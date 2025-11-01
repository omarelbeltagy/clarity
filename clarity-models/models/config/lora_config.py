from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.general_utils import (
    as_int,
    as_float,
    as_bool,
    as_str
)


@dataclass
class LoRAConfig:
    """LoRA-specific configuration."""
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, cfg: Dict) -> "LoRAConfig":
        """Create LoRAConfig from dictionary."""
        return cls(
            r=as_int(cfg.get("r", 8), 8),
            alpha=as_int(cfg.get("alpha", 16), 16),
            dropout=as_float(cfg.get("dropout", 0.05), 0.05),
            task_type=as_str(cfg.get("task_type", "CAUSAL_LM"), "CAUSAL_LM"),
            target_modules=cfg.get("target_modules"),
            bias=as_str(cfg.get("bias", "none"), "none"),
        )


@dataclass
class LoRATrainingConfig:
    """Training hyperparameters."""
    max_length: int = 256
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
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
    dataloader_num_workers: int = 0
    optimizer: str = "adamw_torch"

    @classmethod
    def from_dict(cls, cfg: Dict) -> "LoRATrainingConfig":
        """Create LoRATrainingConfig from dictionary."""
        return cls(
            max_length=as_int(cfg.get("max_length", 256), 256),
            batch_size=as_int(cfg.get("batch_size", 2), 2),
            gradient_accumulation_steps=as_int(cfg.get("gradient_accumulation_steps", 8), 8),
            learning_rate=as_float(cfg.get("learning_rate", 3e-4), 3e-4),
            num_epochs=as_int(cfg.get("num_epochs", 5), 5),
            warmup_ratio=as_float(cfg.get("warmup_ratio", 0.1), 0.1),
            weight_decay=as_float(cfg.get("weight_decay", 0.01), 0.01),
            eval_strategy=as_str(cfg.get("eval_strategy", "epoch"), "epoch"),
            save_strategy=as_str(cfg.get("save_strategy", "epoch"), "epoch"),
            eval_steps=as_int(cfg.get("eval_steps"), None),
            save_steps=as_int(cfg.get("save_steps"), None),
            save_total_limit=as_int(cfg.get("save_total_limit", 3), 3),
            logging_steps=as_int(cfg.get("logging_steps", 5), 5),
            metric_for_best_model=as_str(cfg.get("metric_for_best_model", "eval_loss"), "eval_loss"),
            greater_is_better=as_bool(cfg.get("greater_is_better", False), False),
            dataloader_num_workers=as_int(cfg.get("dataloader_num_workers", 0), 0),
            optimizer=as_str(cfg.get("optimizer", "adamw_torch"), "adamw_torch"),
        )


@dataclass
class LoRADataConfig:
    """Data configuration."""
    train_files: List[str] = None
    valid_files: List[str] = None
    train_sample_size: Optional[int] = None
    valid_sample_size: Optional[int] = None
    # New dynamic field configuration
    label_field: str = "clarity_label"
    question_field: str = "question"
    context_field: str = "context"

    def __post_init__(self):
        if self.train_files is None:
            self.train_files = [
                "/app/data/cleaned/train.json",
                "./data/cleaned/train.json",
                "../data/cleaned/train.json",
                "../../data/cleaned/train.json"
            ]
        if self.valid_files is None:
            self.valid_files = [
                "/app/data/cleaned/valid.json",
                "./data/cleaned/valid.json",
                "../data/cleaned/valid.json",
                "../../data/cleaned/valid.json"
            ]

    @classmethod
    def from_dict(cls, cfg: Dict) -> "LoRADataConfig":
        """Create LoRADataConfig from dictionary."""
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
        if "question_field" in cfg:
            instance.text_field_1 = as_str(cfg["question_field"], "question")
        if "context_field" in cfg:
            instance.text_field_2 = as_str(cfg["context_field"], "context")
        return instance


@dataclass
class LoRAModelConfig:
    """Model configuration."""
    model_name: str = "facebook/opt-1.3b"
    use_8bit: bool = True
    trust_remote_code: bool = True
    output_dir: Optional[str] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir: str = f"./.artifacts/{self.model_name}"

    @classmethod
    def from_dict(cls, cfg: Dict) -> "LoRAModelConfig":
        """Create LoRAModelConfig from dictionary."""
        model_name = as_str(cfg.get("model_name", "facebook/opt-1.3b"), "facebook/opt-1.3b")
        return cls(
            model_name=model_name,
            use_8bit=as_bool(cfg.get("use_8bit", True), True),
            trust_remote_code=as_bool(cfg.get("trust_remote_code", True), True),
            output_dir=cfg.get("output_dir")
        )


@dataclass
class LabelConfig:
    """Label mapping configuration for LoRA models."""
    labels: List[str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]

    @classmethod
    def from_dict(cls, cfg: Dict) -> "LabelConfig":
        """Create LabelConfig from dictionary."""
        labels = cfg.get("labels", ["Clear Reply", "Clear Non-Reply", "Ambivalent"])

        return cls(
            labels=labels,
        )


DEFAULT_PROMPT_TEMPLATE = """
Based on a part of the interview where the interviewer asks a set of questions, classify the type of answer the interviewee provided for the following question.

### Interview Context ###
{context}
### Question ###
{question}

### Label ###
{label}
"""


@dataclass
class PromptConfig:
    """Prompt configuration for LoRA models."""
    template: str = DEFAULT_PROMPT_TEMPLATE

    @classmethod
    def from_dict(cls, cfg: Dict) -> "PromptConfig":
        """Create PromptConfig from dictionary."""
        return cls(
            template=as_str(cfg.get("template", DEFAULT_PROMPT_TEMPLATE),
                            DEFAULT_PROMPT_TEMPLATE)
        )
