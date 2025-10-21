import json
import os
from datasets import load_dataset
from loguru import logger


# Load the dataset
logger.info("Loading dataset...")
logger.info("This may take a while depending on your internet connection.")

ds_train = load_dataset("ailsntua/QEvasion", split="train")
if ds_train is None:
    logger.error("Failed to load the training dataset.")
    raise ValueError("Training dataset is None.")
logger.info("Training dataset loaded successfully.")

ds_test = load_dataset("ailsntua/QEvasion", split="test")
if ds_test is None:
    logger.error("Failed to load the testing dataset.")
    raise ValueError("Testing dataset is None.")
logger.info("Testing dataset loaded successfully.")

# Convert to a list of records
logger.info("Converting datasets to list of records...")
records_train = [row for row in ds_train]
records_test = [row for row in ds_test]

# Save full datasets
logger.info("Saving full datasets to /data/full/...")
os.makedirs("/data/full", exist_ok=True)
with open("/data/full/train.json", "w", encoding="utf-8") as f:
    json.dump(records_train, f, ensure_ascii=False, indent=2)

with open("/data/full/test.json", "w", encoding="utf-8") as f:
    json.dump(records_test, f, ensure_ascii=False, indent=2)

# Reduce datasets to essential fields
logger.info("Reducing datasets to essential fields...")
reduced_train = []
for item in records_train:
    reduced_train.append({
        "question": item["question"],
        "answer": item["interview_answer"],
        "clarity_label": item["clarity_label"]
    })

reduced_test = []
for item in records_test:
    reduced_test.append({
        "question": item["question"],
        "answer": item["interview_answer"]
    })

# Save reduced datasets
logger.info("Saving reduced datasets to /data/reduced/...")
os.makedirs("/data/simple", exist_ok=True)
with open("/data/simple/train.json", "w") as f:
    json.dump(reduced_train, f, ensure_ascii=False, indent=2)

with open("/data/simple/test.json", "w") as f:
    json.dump(reduced_test, f, ensure_ascii=False, indent=2)
