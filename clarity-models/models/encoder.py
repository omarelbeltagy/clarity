"""
Encoder Model Fine-tuning Framework
Supports BERT, RoBERTa, DistilBERT, ALBERT, and other encoder-based models
"""

import atexit
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from data.dto import (
    ClassificationRequest,
)
from datasets import Dataset, DatasetDict
from models.config.encoder_config import (
    EncoderModelConfig,
    EncoderTrainingConfig,
    EncoderDataConfig,
    LabelConfig,
)
from models.config.tensorboard_config import TensorboardConfig
from models.tensorboard_manager import TensorboardManager
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from utils.general_utils import (
    cleanup_checkpoints,
    is_running_in_docker,
    as_int,
    as_float,
    as_bool,
    as_str
)
from utils.logger import logger


# =======================================================================================
# Data Loading Utilities
# =======================================================================================

def load_data_from_files(file_list: List[str], sample_size: Optional[int] = None) -> List[dict]:
    """Load data from first available file in list."""
    for file_path in file_list:
        if os.path.exists(file_path):
            logger.info(f"Loading data from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if sample_size:
                logger.info(f"Using sample size: {sample_size}")
                data = data[:sample_size]

            logger.info(f"Loaded {len(data)} samples")
            return data

    raise FileNotFoundError(f"No data file found in: {file_list}")


# =======================================================================================
# Main Trainer Class
# =======================================================================================

class EncoderTrainer:
    """Modular encoder trainer for sequence classification."""

    def __init__(
            self,
            model_config: EncoderModelConfig,
            training_config: EncoderTrainingConfig = EncoderTrainingConfig(),
            data_config: EncoderDataConfig = EncoderDataConfig(),
            label_config: LabelConfig = LabelConfig(),
            tensorboard_config: TensorboardConfig = TensorboardConfig()
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.label_config = label_config
        self.tensorboard_config = tensorboard_config

        self.model = None
        self.tokenizer = None
        self.tensorboard_manager = None
        self.device = self._detect_device()

        # Update model config with label info
        self.model_config.num_labels = len(self.label_config.labels)

    def _detect_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_tokenizer(self):
        """Load and configure tokenizer."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=self.model_config.trust_remote_code
        )

    def _load_model(self):
        """Load and configure model for sequence classification."""
        logger.info(f"Loading model: {self.model_config.model_name}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name,
            num_labels=self.model_config.num_labels,
            id2label=self.label_config.id2label,
            label2id=self.label_config.label2id,
            trust_remote_code=self.model_config.trust_remote_code
        )

        logger.info(f"Model loaded with {self.model_config.num_labels} labels")

    def _preprocess_function(self, batch):
        """Tokenization function"""
        enc = self.tokenizer(
            batch[self.data_config.context_field],
            batch[self.data_config.question_field],
            truncation=True,
            padding="max_length",
            max_length=self.training_config.max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    def _prepare_datasets(self) -> DatasetDict:
        """Load and prepare datasets."""
        logger.info("Preparing datasets...")

        # Load data
        train_data = load_data_from_files(
            self.data_config.train_files,
            self.data_config.train_sample_size
        )
        valid_data = load_data_from_files(
            self.data_config.valid_files,
            self.data_config.valid_sample_size
        )

        # Convert labels to IDs
        for item in train_data:
            item["label"] = self.label_config.label2id[item[self.data_config.label_field]]

        for item in valid_data:
            item["label"] = self.label_config.label2id[item[self.data_config.label_field]]

        # Create HuggingFace datasets
        dataset = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(valid_data),
        })

        encoded = dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        return encoded

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments."""
        cfg = self.training_config

        args = TrainingArguments(
            output_dir=self.model_config.output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            eval_strategy=cfg.eval_strategy,
            save_strategy=cfg.save_strategy,
            save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model=cfg.metric_for_best_model,
            greater_is_better=cfg.greater_is_better,
            logging_dir=f"{self.model_config.output_dir}/logs",
            logging_strategy="steps",
            logging_steps=cfg.logging_steps,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            dataloader_num_workers=cfg.dataloader_num_workers,
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        # Add eval/save steps if strategy is "steps"
        if cfg.eval_strategy == "steps" and cfg.eval_steps:
            args.eval_steps = cfg.eval_steps
        if cfg.save_strategy == "steps" and cfg.save_steps:
            args.save_steps = cfg.save_steps

        return args

    def train(self):
        """Execute the complete training pipeline."""
        logger.info(f"Starting training with model: {self.model_config.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Labels: {self.label_config.labels}")

        if is_running_in_docker():
            logger.warning(
                "⚠️ Detected running inside a Docker container. "
                "GPU acceleration may not be available."
            )

        try:
            # Start Tensorboard
            self.tensorboard_manager = TensorboardManager(self.tensorboard_config)
            if self.tensorboard_manager.start(f"{self.model_config.output_dir}/logs"):
                atexit.register(self.tensorboard_manager.stop)

            # Load components
            self._load_tokenizer()
            self._load_model()

            # Prepare datasets
            encoded_datasets = self._prepare_datasets()

            # Create training arguments
            training_args = self._create_training_arguments()

            # Prepare callbacks
            callbacks = []
            if self.training_config.early_stopping_patience:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=self.training_config.early_stopping_patience,
                        early_stopping_threshold=self.training_config.early_stopping_threshold
                    )
                )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=encoded_datasets["train"],
                eval_dataset=encoded_datasets["validation"],
                processing_class=self.tokenizer,
                callbacks=callbacks,
            )

            # Train
            logger.info("Starting training...")
            logger.info(f"Total training samples: {len(encoded_datasets['train'])}")
            logger.info(f"Total validation samples: {len(encoded_datasets['validation'])}")
            logger.info(
                f"Effective batch size: {self.training_config.batch_size * self.training_config.gradient_accumulation_steps}"
            )

            if self.tensorboard_manager and self.tensorboard_manager.process:
                logger.info(
                    f"View training progress at: http://localhost:{self.tensorboard_config.port}"
                )

            trainer.train()

            # Save model
            logger.info(f"Saving model to {self.model_config.output_dir}")
            trainer.save_model(self.model_config.output_dir)
            self.tokenizer.save_pretrained(self.model_config.output_dir)

            # Cleanup old checkpoints
            cleanup_checkpoints(output_dir=self.model_config.output_dir)

            logger.info("Training complete!")
            self._print_summary(encoded_datasets)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _print_summary(self, datasets):
        """Print training summary."""
        logger.info("=" * 50)
        logger.info("Training Summary:")
        logger.info(f"Model: {self.model_config.model_name}")
        logger.info(f"Training samples: {len(datasets['train'])}")
        logger.info(f"Validation samples: {len(datasets['validation'])}")
        logger.info(f"Labels: {self.label_config.labels}")
        logger.info(f"Epochs: {self.training_config.num_epochs}")
        logger.info(f"Output directory: {self.model_config.output_dir}")
        logger.info("=" * 50)


# =======================================================================================
# Inference API
# =======================================================================================

class EncoderInferenceAPI:
    """Inference API for trained encoder models."""

    def __init__(
            self,
            model_dir: str,
            label_config: LabelConfig,
            data_config: EncoderDataConfig,
            max_length: int = 128
    ):
        self.model_dir = model_dir
        self.label_config = label_config
        self.data_config = data_config
        self.max_length = max_length

        self.model = None
        self.tokenizer = None

        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load the trained model for inference."""
        logger.info(f"Loading model from {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            num_labels=len(self.label_config.labels),
            id2label=self.label_config.id2label,
            label2id=self.label_config.label2id,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def classify(self, data: ClassificationRequest) -> Dict:
        """Classify a request."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize
        inputs = self.tokenizer(
            data.context,
            data.question,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # Get prediction
        pred_id = int(probs.argmax())
        pred_label = self.label_config.id2label[pred_id]

        return {
            "clarity_label": pred_label,
            "confidence": float(probs[pred_id]),
            "scores": {
                self.label_config.id2label[i]: float(p)
                for i, p in enumerate(probs)
            },
        }


# =======================================================================================
# Loader Function for FastAPI Integration
# =======================================================================================

def load_model(encoder_trainer: EncoderTrainer) -> EncoderInferenceAPI:
    """
    Loader function for FastAPI integration.

    If the trained model doesn't exist, it will train it first.
    """
    # Check if trained model exists
    model_exists = (
            os.path.exists(encoder_trainer.model_config.output_dir) and
            (
                    os.path.exists(os.path.join(encoder_trainer.model_config.output_dir, "pytorch_model.bin")) or
                    os.path.exists(os.path.join(encoder_trainer.model_config.output_dir, "model.safetensors"))
            )
    )

    if not model_exists:
        logger.warning(f"Trained model not found at {encoder_trainer.model_config.output_dir}")
        logger.info("Starting training...")

        try:
            encoder_trainer.train()
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    else:
        logger.info(f"Found trained model at {encoder_trainer.model_config.output_dir}")

    # Load and return API
    api = EncoderInferenceAPI(
        model_dir=encoder_trainer.model_config.output_dir,
        label_config=encoder_trainer.label_config,
        data_config=encoder_trainer.data_config,
        max_length=encoder_trainer.training_config.max_length,
    )
    api.load_model()
    return api
