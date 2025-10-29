from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.general_utils import (
    as_int,
    as_float,
    as_bool,
    as_str
)


@dataclass
class EncoderModelConfig:
    """Encoder model configuration."""
    model_name: str = "roberta-base"
    num_labels: int = 3
    trust_remote_code: bool = True
    output_dir: Optional[str] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir: str = f"./.artifacts/{self.model_name}"

    @classmethod
    def from_dict(cls, cfg: Dict) -> "EncoderModelConfig":
        """Create EncoderModelConfig from dictionary."""
        model_name = as_str(cfg.get("model_name", "roberta-base"), "roberta-base")
        return cls(
            model_name=model_name,
            num_labels=as_int(cfg.get("num_labels", 3), 3),
            trust_remote_code=as_bool(cfg.get("trust_remote_code", True), True),
            output_dir=cfg.get("output_dir")
        )


@dataclass
class EncoderTrainingConfig:
    """Training hyperparameters for encoder models."""
    max_length: int = 128
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_total_limit: int = 5
    logging_steps: int = 5
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0

    @classmethod
    def from_dict(cls, cfg: Dict) -> "EncoderTrainingConfig":
        """Create EncoderTrainingConfig from dictionary."""
        return cls(
            max_length=as_int(cfg.get("max_length", 128), 128),
            batch_size=as_int(cfg.get("batch_size", 16), 16),
            gradient_accumulation_steps=as_int(cfg.get("gradient_accumulation_steps", 1), 1),
            learning_rate=as_float(cfg.get("learning_rate", 2e-5), 2e-5),
            num_epochs=as_int(cfg.get("num_epochs", 3), 3),
            warmup_ratio=as_float(cfg.get("warmup_ratio", 0.1), 0.1),
            weight_decay=as_float(cfg.get("weight_decay", 0.01), 0.01),
            eval_strategy=as_str(cfg.get("eval_strategy", "epoch"), "epoch"),
            save_strategy=as_str(cfg.get("save_strategy", "epoch"), "epoch"),
            eval_steps=as_int(cfg.get("eval_steps"), None),
            save_steps=as_int(cfg.get("save_steps"), None),
            save_total_limit=as_int(cfg.get("save_total_limit", 3), 3),
            logging_steps=as_int(cfg.get("logging_steps", 10), 10),
            metric_for_best_model=as_str(cfg.get("metric_for_best_model", "eval_loss"), "eval_loss"),
            greater_is_better=as_bool(cfg.get("greater_is_better", False), False),
            load_best_model_at_end=as_bool(cfg.get("load_best_model_at_end", True), True),
            early_stopping_patience=as_int(cfg.get("early_stopping_patience"), None),
            early_stopping_threshold=as_float(cfg.get("early_stopping_threshold", 0.0), 0.0),
        )


@dataclass
class EncoderDataConfig:
    """Data configuration for encoder models."""
    train_files: List[str] = None
    valid_files: List[str] = None
    train_sample_size: Optional[int] = None
    valid_sample_size: Optional[int] = None
    label_field: str = "clarity_label"
    text_field_1: str = "question"
    text_field_2: str = "answer"

    def __post_init__(self):
        if self.train_files is None:
            self.train_files = [
                "/app/data/simple/train.json",
                "./data/simple/train.json",
                "../data/simple/train.json",
                "../../data/simple/train.json"
            ]
        if self.valid_files is None:
            self.valid_files = [
                "/app/data/simple/valid.json",
                "./data/simple/valid.json",
                "../data/simple/valid.json",
                "../../data/simple/valid.json"
            ]

    @classmethod
    def from_dict(cls, cfg: Dict) -> "EncoderDataConfig":
        """Create EncoderDataConfig from dictionary."""
        instance = cls()
        if "train_files" in cfg:
            instance.train_files = cfg["train_files"]
        if "valid_files" in cfg:
            instance.valid_files = cfg["valid_files"]
        if "train_sample_size" in cfg:
            instance.train_sample_size = as_int(cfg["train_sample_size"], None)
        if "valid_sample_size" in cfg:
            instance.valid_sample_size = as_int(cfg["valid_sample_size"], None)
        if "label_field" in cfg:
            instance.label_field = as_str(cfg["label_field"], "clarity_label")
        if "text_field_1" in cfg:
            instance.text_field_1 = as_str(cfg["text_field_1"], "question")
        if "text_field_2" in cfg:
            instance.text_field_2 = as_str(cfg["text_field_2"], "answer")
        return instance


@dataclass
class LabelConfig:
    """Label mapping configuration."""
    labels: List[str] = None
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[int, str]] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]

        if self.label2id is None:
            self.label2id = {label: i for i, label in enumerate(self.labels)}

        if self.id2label is None:
            self.id2label = {i: label for label, i in self.label2id.items()}

    @classmethod
    def from_dict(cls, cfg: Dict) -> "LabelConfig":
        """Create LabelConfig from dictionary."""
        labels = cfg.get("labels", ["Clear Reply", "Clear Non-Reply", "Ambivalent"])
        label2id = cfg.get("label2id")
        id2label = cfg.get("id2label")

        # Convert id2label keys to int if provided as strings
        if id2label is not None:
            id2label = {int(k): v for k, v in id2label.items()}

        return cls(
            labels=labels,
            label2id=label2id,
            id2label=id2label
        )
