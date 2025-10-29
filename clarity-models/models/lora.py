"""
LoRA Fine-tuning Framework
"""

import atexit
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

import torch
from models.config.lora_config import (
    LoRAConfig,
    LoRATrainingConfig,
    LoRADataConfig,
    LoRAModelConfig,
    LabelConfig
)
from models.config.tensorboard_config import TensorboardConfig
from models.tensorboard_manager import TensorboardManager
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
from utils.general_utils import (
    is_running_in_docker,
    as_int,
    as_float,
    as_bool,
    as_str
)
from utils.logger import logger


# =======================================================================================
# Default Prompt Formatting Function
# =======================================================================================

def create_default_format_function(data_config):
    """Factory function to create format function with dynamic field names."""

    def format_fn(item: dict) -> str:
        template = """Based on a part of the interview where the interviewer asks a set of questions, classify the type of answer the interviewee provided for the following question.

### Question ###
{field_1}
### Answer ###
{field_2}

### Label ###
{label}"""
        # Use configurable field names
        field_1 = item.get(data_config.text_field_1, "")
        field_2 = item.get(data_config.text_field_2, "")
        label = item.get(data_config.label_field, "")

        return template.format(
            field_1=field_1,
            field_2=field_2,
            label=label
        )

    return format_fn


@dataclass
class PromptConfig:
    """Prompt template configuration."""
    format_function: Optional[Callable] = None


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
            model_config: LoRAModelConfig,
            lora_config: LoRAConfig = LoRAConfig(),
            training_config: LoRATrainingConfig = LoRATrainingConfig(),
            data_config: LoRADataConfig = LoRADataConfig(),
            label_config: LabelConfig = LabelConfig(),
            prompt_config: PromptConfig = PromptConfig(),
            tensorboard_config: TensorboardConfig = TensorboardConfig()
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.training_config = training_config
        self.data_config = data_config
        self.label_config = label_config  # Add this
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
            logger.warning(
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


def create_extraction_function(label_config: LabelConfig):
    """Factory function to create extraction function with dynamic labels."""

    def extract_fn(text: str) -> str:
        return _extract_classification_label(text, valid_labels=label_config.labels)

    return extract_fn


# =======================================================================================
# Inference API
# =======================================================================================

class InferenceAPI:
    """Generic inference API for trained models."""

    def __init__(
            self,
            model_dir: str,
            model_base: str,
            format_function: Optional[Callable],
            extract_function: Optional[Callable[[str], str]]
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

    extract_function = create_extraction_function(
        lora_trainer.label_config)

    # Load and return API
    api = InferenceAPI(
        model_dir=lora_trainer.model_config.output_dir,
        model_base=lora_trainer.model_config.model_name,
        format_function=lora_trainer.prompt_config.format_function,
        extract_function=extract_function
    )
    api.load_model()
    return api
