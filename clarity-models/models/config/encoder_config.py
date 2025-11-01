"""Encoder configuration dataclasses.

This module contains configuration dataclasses used to configure encoder-based
models (e.g. BERT, RoBERTa) for fine-tuning, training hyperparameters, data
locations and label mappings.
"""

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
    """Encoder model configuration.

    Parameters
    ----------
    model_name : str, optional
        Pretrained model identifier (default: "roberta-base").
    num_labels : int, optional
        Number of classification labels (default: 3). This is updated from the
        provided LabelConfig when trainer is initialized.
    trust_remote_code : bool, optional
        Allow model/tokenizer code from the model repository (default: True).
    output_dir : str or None, optional
        Directory to save model artifacts. If None, a default path is created
        from the model_name.

    Notes
    -----
    The `__post_init__` method sets a sensible default for `output_dir` when not
    provided.

    """
    model_name: str = "roberta-base"
    num_labels: int = 3
    trust_remote_code: bool = True
    output_dir: Optional[str] = None

    def __post_init__(self):
        """Finalize configuration after initialization.

        Sets `output_dir` to a default path based on `model_name` if it was not
        supplied by the user.

        """
        if self.output_dir is None:
            self.output_dir: str = f"./.artifacts/{self.model_name}"

    @classmethod
    def from_dict(cls, cfg: Dict) -> "EncoderModelConfig":
        """Create EncoderModelConfig from a dictionary.

        Parameters
        ----------
        cfg : dict
            Dictionary containing configuration keys. Accepted keys include
            "model_name", "num_labels", "trust_remote_code", "output_dir".

        Returns
        -------
        EncoderModelConfig
            An initialized configuration object.

        """
        model_name = as_str(cfg.get("model_name", "roberta-base"), "roberta-base")
        return cls(
            model_name=model_name,
            num_labels=as_int(cfg.get("num_labels", 3), 3),
            trust_remote_code=as_bool(cfg.get("trust_remote_code", True), True),
            output_dir=cfg.get("output_dir")
        )


@dataclass
class EncoderTrainingConfig:
    """Training hyperparameters for encoder models.

    Parameters
    ----------
    max_length : int
        Maximum sequence length for tokenization.
    batch_size : int
        Per-device batch size for training and evaluation.
    gradient_accumulation_steps : int
        Number of steps to accumulate gradients before optimizer step.
    learning_rate : float
        Initial learning rate.
    num_epochs : int
        Number of training epochs.
    warmup_ratio : float
        Fraction of total steps used for linear learning rate warmup.
    weight_decay : float
        Weight decay for optimizer.
    eval_strategy, save_strategy : str
        Strategy keys accepted by HuggingFace TrainingArguments (e.g. "epoch" or "steps").
    eval_steps, save_steps : int or None
        Step intervals for evaluation/checkpointing when strategy is "steps".
    save_total_limit : int
        Max number of checkpoint folders to keep.
    logging_steps : int
        Interval (in steps) for logging.
    metric_for_best_model : str
        Metric name passed to Trainer for selecting best model.
    greater_is_better : bool
        Whether higher metric values are better.
    load_best_model_at_end : bool
        Whether to load the best model at the end of training.
    early_stopping_patience : int or None
        Early stopping patience in evaluation calls (if enabled).
    dataloader_num_workers : int
        Number of worker processes for data loading.

    """
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
    dataloader_num_workers: int = 0

    @classmethod
    def from_dict(cls, cfg: Dict) -> "EncoderTrainingConfig":
        """Create EncoderTrainingConfig from a dictionary.

        Parameters
        ----------
        cfg : dict
            Dictionary with possible keys matching the dataclass fields.

        Returns
        -------
        EncoderTrainingConfig
            Populated training configuration.

        """
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
            dataloader_num_workers=as_int(cfg.get("dataloader_num_workers", 0), 0),
        )


@dataclass
class EncoderDataConfig:
    """Data configuration for encoder models.

    Parameters
    ----------
    train_files : list of str or None
        Candidate paths for training data. First existing path is used.
    valid_files : list of str or None
        Candidate paths for validation data.
    train_sample_size, valid_sample_size : int or None
        If set, only the first N samples are used from the corresponding dataset.
    label_field, context_field, question_field : str
        Field names expected inside JSON records for label, context and question.

    Notes
    -----
    If file lists are None, a set of default relative and absolute paths are
    populated in `__post_init__` to ease running in containerized or local setups.

    """
    train_files: List[str] = None
    valid_files: List[str] = None
    train_sample_size: Optional[int] = None
    valid_sample_size: Optional[int] = None
    label_field: str = "clarity_label"
    context_field: str = "context"
    question_field: str = "question"

    def __post_init__(self):
        """Populate default file paths if none were provided."""
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
    def from_dict(cls, cfg: Dict) -> "EncoderDataConfig":
        """Create EncoderDataConfig from dictionary.

        Parameters
        ----------
        cfg : dict
            Dictionary with optional keys: "train_files", "valid_files",
            "train_sample_size", "valid_sample_size", "label_field",
            "context_field", "question_field".

        Returns
        -------
        EncoderDataConfig
            Config instance populated from the dictionary.

        """
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
        if "context_field" in cfg:
            instance.context_field = as_str(cfg["context_field"], "context")
        if "question_field" in cfg:
            instance.question_field = as_str(cfg["question_field"], "question")
        return instance


@dataclass
class LabelConfig:
    """Label mapping configuration.

    Parameters
    ----------
    labels : list of str or None
        Human-readable labels. Defaults to three clarity labels when None.
    label2id : dict or None
        Mapping from label string to integer id. If None it will be derived.
    id2label : dict or None
        Mapping from integer id to label string. If None it will be derived.

    Notes
    -----
    __post_init__ ensures consistent bidirectional mappings between labels and
    ids. The `from_dict` helper will coerce id keys to integers if they are
    provided as strings.

    """
    labels: List[str] = None
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[int, str]] = None

    def __post_init__(self):
        """Ensure `labels`, `label2id` and `id2label` are populated."""
        if self.labels is None:
            self.labels = ["Clear Reply", "Clear Non-Reply", "Ambivalent"]

        if self.label2id is None:
            self.label2id = {label: i for i, label in enumerate(self.labels)}

        if self.id2label is None:
            self.id2label = {i: label for label, i in self.label2id.items()}

    @classmethod
    def from_dict(cls, cfg: Dict) -> "LabelConfig":
        """Create LabelConfig from dictionary.

        Parameters
        ----------
        cfg : dict
            May contain "labels", "label2id" and/or "id2label". Keys in
            id2label that are strings will be converted to integers.

        Returns
        -------
        LabelConfig
            Populated label configuration.

        """
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
