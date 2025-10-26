"""
LoRA Fine-tuning Framework
"""

import atexit
import json
import os
import subprocess
import sys
import time
import torch
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from typing import Dict, List, Optional, Callable
from utils.general_utils import (
    is_running_in_docker,
    as_int,
    as_float,
    as_bool,
    as_str
)
from utils.logger import logger

# =======================================================================================
# Configuration Classes
# =======================================================================================

OUTPUT_BASE_DIR = "./.artifacts"


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
class TrainingConfig:
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
    def from_dict(cls, cfg: Dict) -> "TrainingConfig":
        """Create TrainingConfig from dictionary."""
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


class DataConfig:
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
    def from_dict(cls, cfg: Dict) -> "DataConfig":
        """Create DataConfig from dictionary."""
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


class ModelConfig:
    """Model configuration."""

    def __init__(self):
        self.output_dir: str = f"{OUTPUT_BASE_DIR}/{self.model_name}"

    model_name: str = "facebook/opt-1.3b"
    use_8bit: bool = True
    trust_remote_code: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        instance = cls()
        instance.model_name = as_str(cfg.get("model_name", instance.model_name), instance.model_name)
        instance.use_8bit = as_bool(cfg.get("use_8bit", instance.use_8bit), instance.use_8bit)
        instance.trust_remote_code = as_bool(
            cfg.get("trust_remote_code", instance.trust_remote_code),
            instance.trust_remote_code
        )
        instance.output_dir = f"{OUTPUT_BASE_DIR}/{instance.model_name}"
        return instance


@dataclass
class TensorboardConfig:
    """Tensorboard configuration."""
    auto_start: bool = True
    port: int = 6006
    host: str = "0.0.0.0"

    @classmethod
    def from_dict(cls, cfg: Dict) -> "TensorboardConfig":
        """Create TensorboardConfig from dictionary."""
        return cls(
            auto_start=as_bool(cfg.get("auto_start", True), True),
            port=as_int(cfg.get("port", 6006), 6006),
            host=as_str(cfg.get("host", "0.0.0.0"), "0.0.0.0"),
        )


# =======================================================================================
# Default Prompt Formatting Function
# =======================================================================================

def _default_format_clarity_prompt(item: dict) -> str:
    template = """Based on a part of the interview where the interviewer asks a set of questions, classify the type of answer the interviewee provided for the following question.

### Question ###
{question}
### Answer ###
{answer}

### Label ###
{label}"""
    # if label is missing, use a placeholder
    if 'clarity_label' not in item:
        item['clarity_label'] = ""
    return template.format(
        question=item['question'],
        answer=item['answer'],
        label=item['clarity_label']
    )


@dataclass
class PromptConfig:
    """Prompt template configuration."""
    format_function: Optional[Callable] = _default_format_clarity_prompt


# =======================================================================================
# Tensorboard Manager
# =======================================================================================

class TensorboardManager:
    """Manages Tensorboard process."""

    def __init__(self, config: TensorboardConfig):
        self.config = config
        self.logdir = None
        self.process = None

    def start(self, logdir: str):
        """Start Tensorboard server."""
        if not self.config.auto_start:
            return False

        self.logdir = logdir

        try:
            logger.info(f"Starting Tensorboard on port {self.config.port}...")

            self.process = subprocess.Popen(
                [
                    "tensorboard",
                    "--logdir", self.logdir,
                    "--port", str(self.config.port),
                    "--host", self.config.host
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            time.sleep(3)

            if self.process.poll() is None:
                url = f"http://localhost:{self.config.port}"
                logger.info(f"Tensorboard started successfully at {url}")
                return True
            else:
                logger.error("Tensorboard failed to start")
                return False

        except FileNotFoundError:
            logger.error("Tensorboard not found.")
            return False
        except Exception as e:
            logger.error(f"Error starting Tensorboard: {e}")
            return False

    def stop(self):
        """Stop Tensorboard server."""
        if self.process and self.process.poll() is None:
            logger.info("Stopping Tensorboard...")
            self.process.terminate()
            self.process.wait()
            logger.info("Tensorboard stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


# =======================================================================================
# Dataset
# =======================================================================================

class GenericDataset(Dataset):
    """Generic dataset for text classification tasks."""

    def __init__(
            self,
            data_files: List[str],
            tokenizer: AutoTokenizer,
            format_function: Callable[[dict], str],
            max_length: int = 256,
            sample_size: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_function = format_function

        # Find and load data
        data_file = self._find_data_file(data_files)
        logger.info(f"Loading data from {data_file}")

        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if sample_size:
            logger.info(f"Using sample size: {sample_size}")
            self.data = self.data[:sample_size]

        logger.info(f"Loaded {len(self.data)} samples")

    @staticmethod
    def _find_data_file(file_list: List[str]) -> str:
        """Find first existing file from list."""
        for file_path in file_list:
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(f"No data file found in: {file_list}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format using provided function
        prompt = self.format_function(item)

        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Create labels
        labels = encodings['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


# =======================================================================================
# Main Trainer Class
# =======================================================================================

class LoRATrainer:
    """Modular LoRA trainer for any causal language model."""

    def __init__(
            self,
            model_config: ModelConfig,
            lora_config: LoRAConfig = LoRAConfig(),
            training_config: TrainingConfig = TrainingConfig(),
            data_config: DataConfig = DataConfig(),
            prompt_config: PromptConfig = PromptConfig(),
            tensorboard_config: TensorboardConfig = TensorboardConfig()
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.training_config = training_config
        self.data_config = data_config
        self.prompt_config = prompt_config
        self.tensorboard_config = tensorboard_config

        self.model = None
        self.tokenizer = None
        self.tensorboard_manager = None
        self.device = self._detect_device()

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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self):
        """Load and configure model."""
        logger.info("Loading base model (this may take a few minutes)...")

        # Configure quantization
        bnb_config = None
        use_8bit_actual = False

        if self.model_config.use_8bit and torch.cuda.is_available():
            logger.info("Using 8-bit quantization on CUDA")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            use_8bit_actual = True
        elif self.model_config.use_8bit:
            logger.warning("8-bit quantization requested but not available")

        # Determine device mapping
        device_map = "auto" if torch.cuda.is_available() else None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            quantization_config=bnb_config if use_8bit_actual else None,
            device_map=device_map,
            trust_remote_code=self.model_config.trust_remote_code,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True
        )

        # Move to device if using MPS or CPU
        if self.device in ["mps", "cpu"]:
            self.model = self.model.to(self.device)
            logger.info(f"Model moved to {self.device}")

        # Prepare model for k-bit training
        if use_8bit_actual:
            self.model = prepare_model_for_kbit_training(self.model)

    def _configure_lora(self):
        """Configure and apply LoRA."""
        logger.info("Configuring LoRA...")

        model_type = self.model.config.model_type
        logger.info(f"Model type: {model_type}")

        # Determine target modules
        if self.lora_config.target_modules:
            target_modules = self.lora_config.target_modules
            logger.info(f"Using custom target modules: {target_modules}")
        elif model_type in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
            target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type]
            logger.info(f"Using default target modules: {target_modules}")
        else:
            target_modules = ["q_proj", "v_proj"]
            logger.info(f"Using fallback target modules: {target_modules}")

        # Create LoRA config
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type,
            target_modules=target_modules
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def _create_datasets(self):
        """Create training and validation datasets."""
        logger.info("Loading datasets...")

        train_dataset = GenericDataset(
            data_files=self.data_config.train_files,
            tokenizer=self.tokenizer,
            format_function=self.prompt_config.format_function,
            max_length=self.training_config.max_length,
            sample_size=self.data_config.train_sample_size
        )

        valid_dataset = GenericDataset(
            data_files=self.data_config.valid_files,
            tokenizer=self.tokenizer,
            format_function=self.prompt_config.format_function,
            max_length=self.training_config.max_length,
            sample_size=self.data_config.valid_sample_size
        )

        return train_dataset, valid_dataset

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments."""
        cfg = self.training_config

        return TrainingArguments(
            output_dir=self.model_config.output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            fp16=self.device != "cpu",
            remove_unused_columns=False,
            report_to=["tensorboard"],
            logging_steps=cfg.logging_steps,
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model=cfg.metric_for_best_model,
            greater_is_better=cfg.greater_is_better,
            weight_decay=cfg.weight_decay,
            dataloader_num_workers=0,
            dataloader_pin_memory=torch.cuda.is_available(),
            optim="adamw_torch",
        )

    def train(self):
        """Execute the complete training pipeline."""
        logger.info(f"Starting training with model: {self.model_config.model_name}")
        logger.info(f"Device: {self.device}")

        if is_running_in_docker():
            logger.warn(
                "⚠️ Detected running inside a Docker container. This will work slowly unless GPU access is properly configured.")

        try:
            # Start Tensorboard
            self.tensorboard_manager = TensorboardManager(self.tensorboard_config)
            if self.tensorboard_manager.start(self.model_config.output_dir):
                atexit.register(self.tensorboard_manager.stop)

            # Load components
            self._load_tokenizer()
            self._load_model()
            self._configure_lora()

            # Create datasets
            train_dataset, valid_dataset = self._create_datasets()

            # Create training arguments
            training_args = self._create_training_arguments()

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
            )

            # Train
            logger.info("Starting training...")
            logger.info(f"Total training samples: {len(train_dataset)}")
            logger.info(f"Total validation samples: {len(valid_dataset)}")
            logger.info(
                f"Effective batch size: {self.training_config.batch_size * self.training_config.gradient_accumulation_steps}"
            )

            if self.tensorboard_manager and self.tensorboard_manager.process:
                logger.info(f"View training progress at: http://localhost:{self.tensorboard_config.port}")

            trainer.train()

            # Save model
            logger.info(f"Saving model to {self.model_config.output_dir}")
            trainer.save_model(self.model_config.output_dir)
            self.tokenizer.save_pretrained(self.model_config.output_dir)

            logger.info("Training complete!")
            self._print_summary(train_dataset, valid_dataset)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _print_summary(self, train_dataset, valid_dataset):
        """Print training summary."""
        logger.info("=" * 50)
        logger.info("Training Summary:")
        logger.info(f"Model: {self.model_config.model_name}")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(valid_dataset)}")
        logger.info(f"Epochs: {self.training_config.num_epochs}")
        logger.info(f"Output directory: {self.model_config.output_dir}")
        logger.info("=" * 50)


def _extract_classification_label(text: str, valid_labels: List[str]) -> str:
    """
    Extract classification label from generated text.

    Args:
        text: Generated text from model
        valid_labels: List of valid classification labels

    Returns:
        Extracted label or first valid label as default
    """
    text_lower = text.strip().lower()

    # Try exact match (case insensitive)
    for label in valid_labels:
        if label.lower() in text_lower:
            return label

    # Try partial match
    for label in valid_labels:
        label_words = label.lower().split()
        if all(word in text_lower for word in label_words):
            return label

    # Try first word match
    first_word = text_lower.split()[0] if text_lower else ""
    for label in valid_labels:
        if first_word in label.lower():
            return label

    # Default to first label with warning
    logger.warning(f"Could not extract label from: '{text}'. Using default: {valid_labels[0]}")
    return valid_labels[0]


def _extract_clarity_label(text: str) -> str:
    """
    Specific extraction function for clarity classification.
    Example of a task-specific extractor.
    """
    return _extract_classification_label(
        text,
        valid_labels=["Clear Reply", "Clear Non-Reply", "Ambivalent"]
    )


# =======================================================================================
# Inference API
# =======================================================================================

class InferenceAPI:
    """Generic inference API for trained models."""

    def __init__(
            self,
            model_dir: str,
            model_base: str,
            format_function: Optional[Callable] = _default_format_clarity_prompt,
            extract_function: Optional[Callable[[str], str]] = _extract_clarity_label
    ):
        self.model_dir = model_dir
        self.model_base = model_base
        self.format_function = format_function
        self.extract_function = extract_function

        self.model = None
        self.tokenizer = None

        # Detect device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load the trained model for inference."""
        logger.info(f"Loading model from {self.model_dir}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                device_map=self.device,
                trust_remote_code=True,
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            if self.model_base:
                logger.error(f"Error loading model: {e}")
                logger.info(f"Attempting to load base model: {self.model_base}")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_base,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_base,
                    device_map=self.device,
                    trust_remote_code=True,
                    dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model.eval()
                logger.warning("Using base model without fine-tuning")
            else:
                raise

    def classify(
            self,
            question: str,
            answer: str,
            max_new_tokens: int = 10,
            temperature: float = 1.0
    ) -> Dict[str, str]:
        """Generate prediction for input text."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input_text = self.format_function({
            "question": question,
            "answer": answer
        })

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )

        # Decode only new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Extract result
        result = self.extract_function(generated_text)

        return {
            "generated_text": generated_text.strip(),
            "extracted_result": result
        }


# =======================================================================================
# Loader Function for FastAPI Integration
# =======================================================================================

def load_model(lora_trainer: LoRATrainer) -> InferenceAPI:
    """
    Loader function for FastAPI integration.
    This is called by the FastAPI app to initialize the model.

    If the trained model doesn't exist, it will train it first.
    """
    # Check if trained model exists
    model_exists = os.path.exists(lora_trainer.model_config.output_dir) and os.path.exists(
        os.path.join(lora_trainer.model_config.output_dir, "adapter_config.json")
    )

    if not model_exists:
        logger.warning(f"Trained model not found at {lora_trainer.model_config.output_dir}")
        logger.info("Starting training...")

        try:
            lora_trainer.train()
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.warning("Will load base model without fine-tuning")
    else:
        logger.info(f"Found trained model at {lora_trainer.model_config.output_dir}")

    # Load and return API
    api = InferenceAPI(
        model_dir=lora_trainer.model_config.output_dir,
        model_base=lora_trainer.model_config.model_name,
        format_function=lora_trainer.prompt_config.format_function,
        extract_function=_extract_clarity_label
    )
    api.load_model()
    return api
