import json
import os
import random
import sys
import yaml
from datasets import load_dataset
from loguru import logger
from cleaning import clean_single_text

DATA_DIR_FULL = "/data/full"
DATA_DIR_SIMPLE = "/data/simple"


def save_json(data, path):
    """Save data as JSON to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reduce_dataset(data, include_label=True):
    """Reduce dataset to essential fields."""
    return [
        {
            "question": item["interview_question"],
            "answer": item["interview_answer"],
            "question_clean": clean_single_text(item["interview_question"], item["president"]),
            "answer_clean":clean_single_text(item["interview_answer"], item["president"]),
            **({"clarity_label": item["clarity_label"]} if include_label else {})
        }
        for item in data
    ]


def main():
    """Main function to load, process, and save datasets."""
    # Load logging configuration from YAML
    with open("logging.yaml", "r") as f:
        log_config = yaml.safe_load(f)
        for handler in log_config.get("handlers", []):
            if handler.get("sink") == "sys.stdout":
                handler["sink"] = sys.stdout
    logger.configure(**log_config)

    logger.info("Loading datasets...")
    ds_train = load_dataset("ailsntua/QEvasion", split="train")
    ds_test = load_dataset("ailsntua/QEvasion", split="test")

    if ds_train is None or ds_test is None:
        raise ValueError("Failed to load dataset(s).")

    logger.info("Splitting train into train/valid...")
    records_train = [row for row in ds_train]
    random.shuffle(records_train)
    split_idx = int(0.8 * len(records_train))
    train_data, valid_data = records_train[:split_idx], records_train[split_idx:]
    test_data = [row for row in ds_test]

    logger.info("Saving full datasets...")
    save_json(train_data, f"{DATA_DIR_FULL}/train.json")
    save_json(valid_data, f"{DATA_DIR_FULL}/valid.json")
    save_json(test_data, f"{DATA_DIR_FULL}/test.json")

    logger.info("Saving reduced datasets...")
    save_json(reduce_dataset(train_data), f"{DATA_DIR_SIMPLE}/train.json")
    save_json(reduce_dataset(valid_data), f"{DATA_DIR_SIMPLE}/valid.json")
    save_json(reduce_dataset(test_data, include_label=False), f"{DATA_DIR_SIMPLE}/test.json")


if __name__ == "__main__":
    main()
