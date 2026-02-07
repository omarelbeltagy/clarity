import os
import re
import random
import logging
from collections import defaultdict, Counter

import nltk
import ssl
import wordninja
from datasets import load_dataset
from rapidfuzz import process
from wordfreq import zipf_frequency, top_n_list


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("words", quiet=True)

from nltk.corpus import words as nltk_words

ENGLISH_WORDS = (
    set(w.lower() for w in nltk_words.words())
    | set(top_n_list("en", 250_000))
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
TEXT_FIELDS = [
    "title",
    "interview_question",
    "interview_answer",
    "gpt3.5_summary",
    "gpt3.5_prediction",
    "question"
]

TOKEN_REGEX = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def extract_tokens(text: str):
    """
    Extract word-like tokens while keeping contractions.
    """
    if not text or not isinstance(text, str):
        return []
    return TOKEN_REGEX.findall(text)


def is_invalid_english_word(token: str) -> bool:
    """
    Check if a token is NOT a valid English word according to NLTK.
    """
    token = token.lower()

    # Ignore very short tokens
    if len(token) <= 1:
        return False

    # Ignore pure numbers
    if token.isdigit():
        return False

    return token not in ENGLISH_WORDS

def strip_possessive(token: str):
    if token.endswith("'s"):
        return token[:-2], "'s"
    return token, ""


def is_valid(word: str) -> bool:
    return zipf_frequency(word, "en") > 3.0


def safe_split(token: str):
    """Split only if ALL parts are valid English words."""
    parts = wordninja.split(token)
    
    # Must split into 2+ parts
    if len(parts) <= 1:
        return None
    
    # ALL parts must be valid
    if not all(is_valid(p) for p in parts):
        return None
    
    # Reject if any part is too short (likely bad split)
    if any(len(p) <= 2 for p in parts):
        return None
    
    return " ".join(parts)


def correct_spelling(token: str):
    """
    Fix spelling by ADDING missing letters only (no removals).
    Prioritizes middle insertions over suffix changes.
    """
    
    candidates = process.extract(
        token,
        ENGLISH_WORDS,
        score_cutoff=75,
        limit=15
    )

    if not candidates:
        return None

    best_candidate = None
    best_score = 0
    
    for candidate, _, _ in candidates:
        #Only accept if candidate is longer
        letters_added = len(candidate) - len(token)
        
        if letters_added < 0:
            continue
        
        if letters_added > 3:  # Too many letters added - REJECT
            continue
        
        # Get word frequency
        freq = zipf_frequency(candidate, "en")
        
        # Only accept common words
        if freq <= 3.0:
            continue
        
        # Middle Insertion Score
        prefix_match = 0
        for i in range(min(len(token), len(candidate))):
            if token[i] == candidate[i]:
                prefix_match += 1
            else:
                break
        
        suffix_match = 0
        for i in range(1, min(len(token), len(candidate)) + 1):
            if token[-i] == candidate[-i]:
                suffix_match += 1
            else:
                break
        
        # Calculate edge preservation ratio
        edge_match = prefix_match + suffix_match
        edge_ratio = edge_match / len(token) if len(token) > 0 else 0
        
        # Score calculation:
        # - Base: word frequency
        # - Bonus: edge preservation (middle insertions score higher)
        # - Penalty: suffix mismatches
        
        score = freq
        
        # Strong bonus for preserving edges (middle insertion)
        if edge_ratio > 0.7:
            score += 3.0
        elif edge_ratio > 0.5:
            score += 1.5
        
        # Extra bonus for preserving suffix (important for word endings)
        suffix_ratio = suffix_match / min(1, len(token))  # Check last char
        if suffix_ratio > 0.75:
            score += 2.0
        elif suffix_ratio > 0.5:
            score += 1.0
        else:
            score -= 1.0  # Penalty for changing suffix
        
        # Prefer fewer insertions when scores are close
        if letters_added > 0:
            score -= (letters_added * 0.2)
        
        if score > best_score and freq > zipf_frequency(token, "en"):
            best_score = score
            best_candidate = candidate
    
    return best_candidate


def fix_token(token: str):
    """Fix token: try spelling first, then splitting."""
    original = token
    token = token.lower()

    base, suffix = strip_possessive(token)

    # 1. Try spelling correction first
    spell_result = correct_spelling(base)
    if spell_result:
        return spell_result + suffix, "spell"

    # 2. Try split only if spelling failed
    split_result = safe_split(base)
    if split_result:
        return split_result + suffix, "split"

    return original, "unchanged"


def analyze_fixes(invalid_counter: Counter):
    before_total = sum(invalid_counter.values())

    fix_stats = Counter()
    fix_map = defaultdict(lambda: {"before": 0, "after": Counter(), "type": None})

    after_invalid_counter = Counter()

    for word, count in invalid_counter.items():
        fixed, fix_type = fix_token(word)

        fix_map[word]["before"] += count
        fix_map[word]["type"] = fix_type
        fix_map[word]["after"][fixed] += count

        # Check if still invalid after fix
        if fixed == word or is_invalid_english_word(fixed.replace(" ", "")):
            after_invalid_counter[word] += count

        fix_stats[fix_type] += count

    after_total = sum(after_invalid_counter.values())

    return before_total, after_total, fix_map, fix_stats




# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
def flag_invalid_words(dataset, split_name: str):
    """
    Iterate through dataset and flag invalid English words.
    """
    logger.info(f"Scanning {split_name} split...")

    invalid_word_counter = Counter()
    invalid_word_examples = defaultdict(list)

    for idx, record in enumerate(dataset):
        for field in TEXT_FIELDS:
            text = record.get(field)
            if not text:
                continue

            tokens = extract_tokens(text)

            for token in tokens:
                if is_invalid_english_word(token):
                    invalid_word_counter[token.lower()] += 1

                    # Store up to 3 examples per word
                    if len(invalid_word_examples[token.lower()]) < 3:
                        invalid_word_examples[token.lower()].append({
                            "split": split_name,
                            "record_index": idx,
                            "field": field,
                            "token": token,
                            "context": text[:200] + ("..." if len(text) > 200 else "")
                        })

    return invalid_word_counter, invalid_word_examples


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------
def main():
    logger.info("Loading QEvasion datasets...")

    ds_train = load_dataset("ailsntua/QEvasion", split="train")
    ds_test = load_dataset("ailsntua/QEvasion", split="test")

    # Scan datasets
    train_counts, train_examples = flag_invalid_words(ds_train, "train")
    test_counts, test_examples = flag_invalid_words(ds_test, "test")

    # Merge results
    all_counts = train_counts + test_counts
    all_examples = {**train_examples, **test_examples}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    logger.info("\n========= WORD FIXES ANALYSIS =========\n")

    before_total, after_total, fix_map, fix_stats = analyze_fixes(all_counts)

    logger.info(f"Invalid words BEFORE fixing: {before_total}")
    logger.info(f"Invalid words AFTER fixing:  {after_total}\n")

    logger.info("Fix type distribution:")
    for k, v in fix_stats.items():
        logger.info(f"  {k:<10} {v}")
    logger.info("")

    logger.info("Top 30 invalid words (before → after):\n")

    for word, count in all_counts.most_common(30):
        info = fix_map[word]
        after_forms = ", ".join(
            f"{k} ({v})" for k, v in info["after"].items()
        )

        logger.info(
            f"{word:<15} {info['before']:>6}  →  "
            f"{after_forms:<20}  [{info['type']}]"
        )

    logger.info("\n========== END ANALYSIS ==========")



# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
