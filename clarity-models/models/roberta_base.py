import json
import os
import random
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

MODEL_DIR = "./.artifacts/roberta-base"
DATA_TRAIN_FILES = [
    "/app/data/simple/train.json",
    "/app/data/simple/train.json",
    "./data/simple/train.json",
    "../data/simple/train.json",
    "../../data/simple/train.json"
]
DATA_VALID_FILES = [
    "/app/data/simple/valid.json",
    "/app/data/simple/valid.json",
    "./data/simple/valid.json",
    "../data/simple/valid.json",
    "../../data/simple/valid.json"
]
MODEL_NAME = "roberta-base"

LABEL2ID = {
    "Clear Reply": 0,
    "Clear Non-Reply": 1,
    "Ambivalent": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class RobertaAPI:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if not (os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin"))
                or os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))):
            self.train()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

    def train(self):
        train_data = []
        for file_path in DATA_TRAIN_FILES:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    train_data = json.load(f)
                break

        if not train_data:
            raise RuntimeError(f"No training data found in {DATA_TRAIN_FILES}")

        valid_data = []
        for file_path in DATA_VALID_FILES:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    valid_data = json.load(f)
                break
        if not valid_data:
            raise RuntimeError(f"No validation data found in {DATA_VALID_FILES}")

        for d in train_data:
            d["label"] = LABEL2ID[d["clarity_label"]]
            del d["clarity_label"]
        for d in valid_data:
            d["label"] = LABEL2ID[d["clarity_label"]]
            del d["clarity_label"]

        dataset = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(valid_data),
        })

        def preprocess(batch):
            enc = self.tokenizer(
                batch["question"],
                batch["answer"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            enc["labels"] = batch["label"]
            return enc

        encoded = dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        args = TrainingArguments(
            output_dir=MODEL_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            load_best_model_at_end=True,
            remove_unused_columns=False,
            logging_dir="./logs",
            logging_strategy="steps",
            logging_steps=10,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=encoded["train"],
            eval_dataset=encoded["validation"],
            processing_class=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(MODEL_DIR)
        self.tokenizer.save_pretrained(MODEL_DIR)

    def classify(self, question, answer):
        inputs = self.tokenizer(
            question,
            answer,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_id = int(probs.argmax())
        pred_label = ID2LABEL[pred_id]
        return {
            "clarity_label": pred_label,
            "confidence": float(probs[pred_id]),
            "scores": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        }


def load_model():
    return RobertaAPI()
