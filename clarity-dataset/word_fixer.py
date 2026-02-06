import os
import re
import random
import logging
from collections import defaultdict, Counter

import nltk
import wordfreq
import ssl
from datasets import load_dataset

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

# Download required NLTK resources (safe to call multiple times)
nltk.download("words", quiet=True)

from nltk.corpus import words as nltk_words

ENGLISH_WORDS = (
    set(w.lower() for w in nltk_words.words())
    | set(wordfreq.top_n_list("en", 500_000))
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
    logger.info("\n========== INVALID ENGLISH WORDS REPORT ==========\n")

    logger.info(f"Total unique invalid words: {len(all_counts)}")
    logger.info(f"Total invalid word occurrences: {sum(all_counts.values())}\n")

    logger.info("Top 50 invalid words by frequency:\n")

    for word, count in all_counts.most_common(50):
        logger.info(f"{word:<20} {count}")

        for ex in all_examples.get(word, []):
            logger.info(
                f"  └─ [{ex['split']} | idx={ex['record_index']} | {ex['field']}] "
                f"...{ex['token']}..."
            )

        logger.info("")

    logger.info("========== END REPORT ==========")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
