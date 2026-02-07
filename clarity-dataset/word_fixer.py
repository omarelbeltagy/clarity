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
    return zipf_frequency(word, "en") > 2.5


def safe_split(token: str):
    parts = wordninja.split(token)
    if len(parts) > 1 and all(is_valid(p) for p in parts):
        return " ".join(parts)
    return None


def correct_spelling(token: str):
    """Try to fix spelling. Returns None if no good correction found."""
    
    max_edits = min(3, len(token) // 3)
    
    candidates = process.extract(
        token,
        ENGLISH_WORDS,
        score_cutoff=80,
        limit=5
    )

    if not candidates:
        return None

    best_candidate = None
    best_freq = 0
    
    for candidate, score, _ in candidates:        
        if abs(len(token) - len(candidate)) > max_edits:
            continue  # Too many letters added/removed
        
        freq = zipf_frequency(candidate, "en")
        
        # Only accept if it's a valid common word
        if freq > 2.0 and freq > best_freq:
            best_candidate = candidate
            best_freq = freq
    
    if best_candidate and best_freq > zipf_frequency(token, "en"):
        return best_candidate
    
    return None


def fix_token(token: str):
    original = token
    token = token.lower()

    base, suffix = strip_possessive(token)

    # Try split first
    split_result = safe_split(base)
    if split_result:
        return split_result + suffix, "split"

    # Try spelling correction
    spell_result = correct_spelling(base)
    if spell_result:
        return spell_result + suffix, "spell"

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
