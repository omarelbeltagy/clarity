from dataclasses import dataclass, field
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
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3
    logging_steps: int = 5
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

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
            eval_steps=as_int(cfg.get("eval_steps", 50), 50),
            save_steps=as_int(cfg.get("save_steps", 50), 50),
            save_total_limit=as_int(cfg.get("save_total_limit", 3), 3),
            logging_steps=as_int(cfg.get("logging_steps", 5), 5),
            metric_for_best_model=as_str(cfg.get("metric_for_best_model", "eval_loss"), "eval_loss"),
            greater_is_better=as_bool(cfg.get("greater_is_better", False), False),
        )


class LoRADataConfig:
    """Data configuration."""
    train_files: List[str] = [
        "/app/data/simple/train.json",
        "./data/simple/train.json",
        "../data/simple/train.json",
        "../../data/simple/train.json"
    ]
    valid_files: List[str] = [
        "/app/data/simple/valid.json",
        "./data/simple/valid.json",
        "../data/simple/valid.json",
        "../../data/simple/valid.json"
    ]
    train_sample_size: Optional[int] = None
    valid_sample_size: Optional[int] = None

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
