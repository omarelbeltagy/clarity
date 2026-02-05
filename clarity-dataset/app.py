"""Dataset utilities for Clarity models and conversion helpers for Together API.

This module provides functions to load, clean and persist datasets used by the
Clarity project, and to convert records to the JSONL format expected by the
Together API for few-shot/zero-shot inference.
"""

import argparse
import json
import os
import random
import sys

import yaml
from datasets import load_dataset
from spacy_cleaner import SpacyCleaner, create_cleaner


from clean import (
    clean_single_text,
    _normalize,
    remove_brackets,
    remove_fillers,
    remove_names
)
#from utils.logger import logger
from loguru import logger


def get_data_path():
    """Return and ensure the data directory exists.

    The function checks the DARA_DIR environment variable and falls back to
    "../data" if not set. The directory will be created if it does not yet
    exist. This ensures that it will work both natively and within Docker.

    Returns
    -------
    str
        Path to the data directory that exists after the call.

    Notes
    -----
    The returned path can be relative or absolute depending on the environment.
    """
    data_path_env = os.getenv("DARA_DIR")

    if data_path_env:
        data_path_dir = data_path_env
    else:
        data_path_dir = "data"

    os.makedirs(data_path_dir, exist_ok=True)

    return data_path_dir


def load_lora_prompt():
    """Load LoRA prompt template from a YAML file.

    The path is determined by the LORA_PROMPT_FILE environment variable or by
    the default path "../assets/prompts/lora.yaml". The YAML file is expected
    to contain a top-level "prompt" key.

    Returns
    -------
    str
        The prompt template string. Returns an empty string if the key is missing.

    Raises
    ------
    FileNotFoundError
        If the resolved prompt file does not exist.
    """
    path = os.getenv("LORA_PROMPT_FILE", "../assets/prompts/lora.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f).get("prompt", "")


def extract_together_entry(item, prompt: str):
    """Create a Together-compatible single example from a dataset item.

    The function composes a prompt by formatting the provided prompt template
    with 'context' and 'question' fields extracted from the input item. It
    returns None if required fields are missing or empty.

    Parameters
    ----------
    item : Mapping
        A mapping-like object representing a dataset record. Expected keys:
        - "interview_question", "interview_answer", "question", "clarity_label"
    prompt : str
        Prompt template containing placeholders for 'context' and 'question'
        (e.g. "...{context}...{question}...").

    Returns
    -------
    dict or None
        If successful, returns a dict with keys:
        - "prompt": formatted prompt string
        - "completion": completion string (e.g. "Label: <label>")
        Returns None if any required field is missing or empty.
    """
    interview_question, interview_answer = item.get("interview_question", "").strip(), item.get("interview_answer",
                                                                                                "").strip()
    question = item.get("question", "").strip()
    label = item.get("clarity_label", "").strip()
    if not (interview_question and interview_answer and question and label):
        return None
    context = f"{interview_question}\n{interview_answer}"
    formatted = prompt.strip().format(context=context, question=question)
    return {"prompt": formatted, "completion": f"Label: {label}"}


def convert_for_together(train_data, valid_data, base_dir):
    """Convert train and validation datasets to Together JSONL format.

    Writes two files under "{base_dir}/together/":
    - train.jsonl
    - valid.jsonl

    Parameters
    ----------
    train_data : Sequence[Mapping]
        Sequence of raw training records.
    valid_data : Sequence[Mapping]
        Sequence of raw validation records.
    base_dir : str
        Base directory where the 'together' subfolder will be created.

    Returns
    -------
    None

    Notes
    -----
    If either train_data or valid_data is empty, the function logs an error and
    returns without writing files. The function uses `load_lora_prompt` to
    obtain the prompt template.
    """
    if not (train_data and valid_data):
        logger.error("Train or validation data empty. Aborting conversion.")
        return

    prompt = load_lora_prompt()
    tgt_dir = os.path.join(base_dir, "together")
    write_jsonl(os.path.join(tgt_dir, "train.jsonl"), train_data, prompt)
    write_jsonl(os.path.join(tgt_dir, "valid.jsonl"), valid_data, prompt)


def write_jsonl(path, data, prompt):
    """Write dataset records to a JSONL file in Together format.

    Each line in the output file corresponds to a JSON object created by
    `extract_together_entry` using the provided prompt template.

    Parameters
    ----------
    path : str
        Destination file path for the JSONL output.
    data : Sequence[Mapping]
        Sequence of raw dataset records.
    prompt : str
        Prompt template for formatting each record.
    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            entry = extract_together_entry(item, prompt)
            if entry:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_json(data, path):
    """Save a Python object as a JSON file.

    Creates parent directories where necessary and writes the JSON file using
    UTF-8 encoding.

    Parameters
    ----------
    data : Any
        JSON-serializable Python object to persist.
    path : str
        Destination file path.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_dataset(data, include_label=True, clean_fillers=False, clean_names=False):
    """Return a reduced and cleaned representation of dataset records using SpaCy cleaner."""
    
    # Create the cleaner
    cleaner = create_cleaner(preserve_negation=True)
    
    result = []
    for item in data:
        question = item["question"]
        context = item["interview_question"] + "\n" + item["interview_answer"]
        president = item.get("president")
        
        # Determine what cleaning to apply
        if clean_fillers or clean_names:
            president_name = president if clean_names else None
            question_clean = cleaner.clean_text(question, remove_stopwords=False, president_name=president_name)
            context_clean = cleaner.clean_text(context, remove_stopwords=False, president_name=president_name)
        else:
            # No cleaning, just normalize punctuation
            question_clean = cleaner._clean_punctuation(question)
            context_clean = cleaner._clean_punctuation(context)
        
        entry = {
            "question_clean": question_clean,
            "context_clean": context_clean,
            "question": question,
            "context": context,
        }
        
        if include_label:
            entry["clarity_label"] = item["clarity_label"]
        
        result.append(entry)
    
    return result


def display_sample(records, title, sample_size):
    """Display random samples from the dataset."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    
    samples = random.sample(records, sample_size)
    
    for i, record in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Question (original): {record.get('question', 'N/A')}")
        if 'question_clean' in record:
            print(f"Question (cleaned):  {record.get('question_clean', 'N/A')}")
        print()


def main():
    """Load, process and save QEvasion datasets.

    The function performs the following steps:
    1. Load the 'ailsntua/QEvasion' dataset splits (train and test).
    2. Shuffle and split the train split into train/validation (80/20).
    3. Persist the full and cleaned datasets under the DARA_DIR directory.
    4. Convert train/valid to Together JSONL format.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any dataset split cannot be loaded.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and clean QEvasion dataset")
    parser.add_argument("--clean-fillers", action="store_true",
                        help="Remove filler words from text")
    parser.add_argument("--clean-names", action="store_true",
                        help="Remove president names from text")
    parser.add_argument("--clean-all", action="store_true",
                        help="Apply all cleaning (fillers + names)")


    args = parser.parse_args()

    # Handle --clean-all flag
    if args.clean_all:
        args.clean_fillers = True
        args.clean_names = True

    # Load logging configuration from YAML
    with open("logging.yaml", "r") as f:
        log_config = yaml.safe_load(f)
        for handler in log_config.get("handlers", []):
            if handler.get("sink") == "sys.stdout":
                handler["sink"] = sys.stdout
    logger.configure(**log_config)

    logger.info("Configuration:")
    logger.info(f"  - Clean fillers: {args.clean_fillers}")
    logger.info(f"  - Clean names: {args.clean_names}")

    logger.info("Loading QEvasion datasets...")
    ds_train = load_dataset("ailsntua/QEvasion", split="train")
    ds_test = load_dataset("ailsntua/QEvasion", split="test")

    if ds_train is None or ds_test is None:
        raise ValueError("Failed to load dataset(s).")

    logger.info("Splitting train into train/valid...")
    records_train = [row for row in ds_train]
    random.seed(42)
    random.shuffle(records_train)

    split_idx = int(0.8 * len(records_train))
    train_data, valid_data = records_train[:split_idx], records_train[split_idx:]
    test_data = [row for row in ds_test]

    data_dir = get_data_path()
    full_dir = os.path.join(data_dir, "full")
    clean_dir = os.path.join(data_dir, "cleaned")

    logger.info("Saving raw datasets...")
    for name, data in {"train": train_data, "valid": valid_data, "test": test_data}.items():
        save_json(data, os.path.join(full_dir, f"{name}.json"))


    # Sample data
    random.seed(90)
    sample_indices = random.sample(range(len(train_data)), 10)
    sample_records_before = [train_data[i] for i in sample_indices]

    # Cleaned sample
    sample_records_after = clean_dataset(
        sample_records_before, 
        clean_fillers=args.clean_fillers,
        clean_names=args.clean_names
    )
    logger.info("\nDisplaying samples AFTER preprocessing...")
    display_sample(sample_records_after, "BEFORE/AFTER PREPROCESSING", 10)

    train_cleaned = clean_dataset(train_data, clean_fillers=args.clean_fillers,
                                   clean_names=args.clean_names)
    valid_cleaned = clean_dataset(valid_data, clean_fillers=args.clean_fillers,
                                   clean_names=args.clean_names)
    test_cleaned = clean_dataset(test_data, clean_fillers=args.clean_fillers,
                                  clean_names=args.clean_names)

    logger.info("Saving cleaned datasets...")
    for name, data in {"train": train_cleaned, "valid": valid_cleaned, "test": test_cleaned}.items():
        save_json(data, os.path.join(clean_dir, f"{name}.json"))

    logger.info("Converting datasets for Together format...")
    convert_for_together(train_data, valid_data, data_dir)
    logger.info("Data processing complete.")


if __name__ == "__main__":
    main()
