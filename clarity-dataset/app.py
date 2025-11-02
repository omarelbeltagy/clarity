import argparse
import json
import os
import random
import sys

import yaml
from datasets import load_dataset
from loguru import logger

from clean import (
    clean_single_text,
    _normalize,
    remove_brackets,
    remove_fillers,
    remove_names
)
from summarize import generate_bert_summary

DATA_DIR_FULL = "/data/full"
DATA_DIR_SIMPLE = "/data/cleaned"


def save_json(data, path):
    """Save data as JSON to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_dataset(data, include_label=True, clean_fillers=False, clean_names=False):
    """
    Clean dataset with configurable cleaning options.
    
    Args:
        data: Dataset to clean
        include_label: Whether to include clarity labels
        clean_fillers: Whether to remove filler words
        clean_names: Whether to remove names
        
    Returns:
        List of cleaned dataset items
    """
    result = []
    
    for item in data:
        question = item["question"]
        context = item["interview_question"] + "\n" + item["interview_answer"]
        president = item["president"]
        
        # Determine which cleaning to apply
        if clean_fillers and clean_names:
            # Use full cleaning function
            question_clean = clean_single_text(question, president)
            context_clean = clean_single_text(context, president)
        elif clean_fillers or clean_names:
            # Partial cleaning
            question_clean = _normalize(question)
            context_clean = _normalize(context)
            
            if clean_names:
                name = [president] if president else []
                question_clean = remove_names(question_clean, name, aggressive_lastname=False)
                context_clean = remove_names(context_clean, name, aggressive_lastname=False)
            
            if clean_fillers:
                question_clean = remove_brackets(question_clean)
                question_clean = remove_fillers(question_clean)
                context_clean = remove_brackets(context_clean)
                context_clean = remove_fillers(context_clean)
            
            question_clean = _normalize(question_clean)
            context_clean = _normalize(context_clean)
        else:
            # No cleaning
            question_clean = question
            context_clean = context
        
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


def main():
    """Main function to load, process, and save datasets."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and clean QEvasion dataset")
    parser.add_argument("--clean-fillers", action="store_true", 
                       help="Remove filler words from text")
    parser.add_argument("--clean-names", action="store_true",
                       help="Remove president names from text")
    parser.add_argument("--clean-all", action="store_true",
                       help="Apply all cleaning (fillers + names)")
    parser.add_argument("--use-bert", action="store_true",
                       help="Generate BERT-based summaries")
    
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
    logger.info(f"  - Use BERT: {args.use_bert}")
    
    logger.info("Loading datasets...")
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

    logger.info("Saving full datasets...")
    save_json(train_data, f"{DATA_DIR_FULL}/train.json")
    save_json(valid_data, f"{DATA_DIR_FULL}/valid.json")
    save_json(test_data, f"{DATA_DIR_FULL}/test.json")

    logger.info("Cleaning datasets...")
    train_cleaned = clean_dataset(train_data, clean_fillers=args.clean_fillers, 
                                   clean_names=args.clean_names)
    valid_cleaned = clean_dataset(valid_data, clean_fillers=args.clean_fillers,
                                   clean_names=args.clean_names)
    test_cleaned = clean_dataset(test_data, clean_fillers=args.clean_fillers,
                                  clean_names=args.clean_names)

    # Apply BERT summaries if requested (after cleaning)
    if args.use_bert:
        logger.info("Generating BERT summaries for train set...")
        train_cleaned[:10] = generate_bert_summary(train_cleaned[:10])

    logger.info("Saving processed datasets...")
    save_json(train_cleaned, f"{DATA_DIR_SIMPLE}/train.json")
    save_json(valid_cleaned, f"{DATA_DIR_SIMPLE}/valid.json")
    save_json(test_cleaned, f"{DATA_DIR_SIMPLE}/test.json")
    
    logger.info("Dataset processing complete!")


if __name__ == "__main__":
    main()
