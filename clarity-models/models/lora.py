"""
LoRA Fine-tuning Framework.

Contains utilities to format prompts, datasets and training/inference
wrappers for LoRA (PEFT) fine-tuning of causal language models.
"""

import atexit
import json
import os
from typing import Dict, List, Optional

import torch
from dto.dto import (
    ClassificationRequest,
)
from models.config.lora_config import (
    LoRAConfig,
    LoRATrainingConfig,
    LoRADataConfig,
    LoRAModelConfig,
    LabelConfig,
    PromptConfig
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
    cleanup_checkpoints,
    is_running_in_docker,
    as_int,
    as_float,
    as_bool,
    as_str
)
from utils.logger import logger


# =======================================================================================
# Prompt Formatting Function
# =======================================================================================

def format_prompt(data_config: LoRADataConfig, prompt_config: PromptConfig, item: Dict) -> str:
    """Format a prompt string from a data item using a template.

    Parameters
    ----------
    data_config : LoRADataConfig
        Configuration that contains the field names to read from `item`.
    prompt_config : PromptConfig
        Template configuration containing `template` with placeholders.
    item : dict
        Single data record containing fields such as question/context/label.

    Returns
    -------
    str
        The formatted prompt ready for tokenization.

    """
    template = prompt_config.template
    question_field = item.get(data_config.question_field, "")
    context_field = item.get(data_config.context_field, "")
    label_field = item.get(data_config.label_field, "")

    prompt = template.format(
        question=question_field,
        context=context_field,
        label=label_field
    )

    return prompt


# =======================================================================================
# Extract label function
# =======================================================================================

def extract_label(label_config: LabelConfig, text: str) -> str:
    """Extract the intended label from generated model text.

    The function tries several heuristics:
    1. Case-insensitive exact match of any label inside the text.
    2. Partial match by checking that all words of a label appear.
    3. First-word match as a fallback.
    4. Default to the first available label with a warning.

    Parameters
    ----------
    label_config : LabelConfig
        Label configuration containing the list `labels`.
    text : str
        Generated text to parse for a label.

    Returns
    -------
    str
        One of the labels from `label_config.labels`.

    """
    valid_labels = label_config.labels
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


# =======================================================================================
# Dataset
# =======================================================================================

class GenericDataset(Dataset):
    """Generic PyTorch Dataset for prompt-based causal LM training.

    Tokenizes prompts constructed from JSON records and prepares labels for
    causal language modeling by masking padding token ids with -100.

    Parameters
    ----------
    data_files : list of str
        Candidate file paths where JSON data is stored. The first existing file
        will be loaded.
    tokenizer : transformers.AutoTokenizer
        Tokenizer used to convert prompt text to model input ids.
    prompt_config : PromptConfig, optional
        Prompt template configuration.
    data_config : LoRADataConfig, optional
        Data field mappings used by format_prompt.
    max_length : int, optional
        Maximum token length for examples.
    sample_size : int or None, optional
        If set, only the first `sample_size` examples will be used.

    """
    def __init__(
            self,
            data_files: List[str],
            tokenizer: AutoTokenizer,
            prompt_config: PromptConfig = PromptConfig(),
            data_config: LoRADataConfig = LoRADataConfig(),
            max_length: int = 256,
            sample_size: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_config = data_config
        self.prompt_config = prompt_config

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
        """Find first existing file path from the candidate list.

        Raises
        ------
        FileNotFoundError
            When no file in `file_list` exists.

        """
        for file_path in file_list:
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(f"No data file found in: {file_list}")

    def __len__(self):
        """Return number of loaded examples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return tokenized input and labels for index `idx`.

        Returns a dictionary with keys 'input_ids', 'attention_mask' and 'labels'
        where label positions corresponding to pad tokens are set to -100.

        """
        item = self.data[idx]

        # Format using provided function
        prompt = format_prompt(
            data_config=self.data_config,
            prompt_config=self.prompt_config,
            item=item
        )

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
    """Modular LoRA trainer for causal language models.

    Coordinates tokenizer/model loading, optional 8-bit quantization setup,
    applying LoRA adapters (PEFT), dataset creation and Trainer-based training.

    Parameters
    ----------
    model_config : LoRAModelConfig
        Base model configuration.
    lora_config : LoRAConfig, optional
        LoRA hyperparameters (r, alpha, dropout, etc.).
    training_config : LoRATrainingConfig, optional
        Training hyperparameters.
    data_config : LoRADataConfig, optional
        Data locations and field names.
    label_config : LabelConfig, optional
        Labels used for extraction/evaluation.
    prompt_config : PromptConfig, optional
        Prompt template config.
    tensorboard_config : TensorboardConfig, optional
        Tensorboard server configuration.

    """
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
        """Detect best device available ("cuda", "mps" or "cpu")."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_tokenizer(self):
        """Load tokenizer and ensure pad token is defined."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=self.model_config.trust_remote_code
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self):
        """Load pretrained causal LM and prepare for (k-)bit training.

        If `use_8bit` is True and CUDA is available, quantization with
        BitsAndBytesConfig is configured and prepare_model_for_kbit_training
        is invoked.

        """
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
        """Create and apply LoRA PEFT configuration to the base model.

        Determines target modules either from user config, a known mapping or
        a reasonable fallback.

        """
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
        """Create training and validation datasets using GenericDataset.

        Returns
        -------
        tuple
            (train_dataset, valid_dataset)

        """
        logger.info("Loading datasets...")

        train_dataset = GenericDataset(
            data_files=self.data_config.train_files,
            tokenizer=self.tokenizer,
            max_length=self.training_config.max_length,
            sample_size=self.data_config.train_sample_size
        )

        valid_dataset = GenericDataset(
            data_files=self.data_config.valid_files,
            tokenizer=self.tokenizer,
            max_length=self.training_config.max_length,
            sample_size=self.data_config.valid_sample_size
        )

        return train_dataset, valid_dataset

    def _create_training_arguments(self) -> TrainingArguments:
        """Construct TrainingArguments from training_config."""
        cfg = self.training_config

        return TrainingArguments(
            output_dir=self.model_config.output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
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
            eval_strategy=cfg.eval_strategy,
            save_strategy=cfg.save_strategy,
            dataloader_num_workers=cfg.dataloader_num_workers,
            dataloader_pin_memory=torch.cuda.is_available(),
            optim=cfg.optimizer,
        )

    def train(self):
        """Run the full training pipeline.

        Starts tensorboard, loads components, applies LoRA adapters,
        runs Trainer.train(), saves artifacts and cleans up checkpoints.

        """
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

            # Cleanup old checkpoints
            cleanup_checkpoints(output_dir=self.model_config.output_dir)

            logger.info("Training complete!")
            self._print_summary(train_dataset, valid_dataset)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _print_summary(self, train_dataset, valid_dataset):
        """Log a brief training summary including sample counts and epochs."""
        logger.info("=" * 50)
        logger.info("Training Summary:")
        logger.info(f"Model: {self.model_config.model_name}")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(valid_dataset)}")
        logger.info(f"Epochs: {self.training_config.num_epochs}")
        logger.info(f"Output directory: {self.model_config.output_dir}")
        logger.info("=" * 50)


# =======================================================================================
# Inference API
# =======================================================================================

class InferenceAPI:
    """Generic inference API for LoRA-trained models.

    Loads a fine-tuned adapter model if available; otherwise attempts to
    load the base model. Provides a `classify` method that returns generated
    text and a label extracted by heuristics.

    Parameters
    ----------
    model_dir : str
        Directory containing the fine-tuned model (or adapter).
    model_base : str
        Name of the base model to fall back to if loading fails.
    label_config : LabelConfig, optional
        Label list to be used for extraction.
    prompt_config : PromptConfig, optional
        Prompt template configuration.
    data_config : LoRADataConfig, optional
        Field mappings for constructing prompts.

    """
    def __init__(
            self,
            model_dir: str,
            model_base: str,
            label_config: LabelConfig = LabelConfig(),
            prompt_config: PromptConfig = PromptConfig(),
            data_config: LoRADataConfig = LoRADataConfig()
    ):
        self.model_dir = model_dir
        self.model_base = model_base
        self.label_config = label_config
        self.prompt_config = prompt_config
        self.data_config = data_config

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
        """Load tokenizer and model for inference.

        Tries to load the fine-tuned artifacts from `model_dir`. If that
        fails and `model_base` is provided, will load the base model instead.

        Raises
        ------
        Exception
            Propagates loader exception if no fallback is available.

        """
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
            data: ClassificationRequest,
            max_new_tokens: int = 10,
            temperature: float = 1.0
    ) -> Dict[str, str]:
        """Generate model output for a single input and extract a label.

        Parameters
        ----------
        data : ClassificationRequest
            DTO containing the fields used by the prompt template.
        max_new_tokens : int, optional
            Maximum number of new tokens to generate.
        temperature : float, optional
            Sampling temperature (not used with deterministic generation settings).

        Returns
        -------
        dict
            {"generated_text": <str>, "extracted_result": <label_str>}

        Raises
        ------
        RuntimeError
            If the model/tokenizer are not loaded.

        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input_text = format_prompt(
            data_config=self.data_config,
            prompt_config=self.prompt_config,
            item=data.dict()
        )

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
        result = extract_label(
            label_config=self.label_config,
            text=generated_text
        )

        return {
            "generated_text": generated_text.strip(),
            "name": result
        }


# =======================================================================================
# Loader Function for FastAPI Integration
# =======================================================================================

def load_model(lora_trainer: LoRATrainer) -> InferenceAPI:
    """Loader for FastAPI integration.

    If a fine-tuned LoRA adapter isn't present, training is invoked. On
    failure the base model is loaded and a warning is issued.

    Parameters
    ----------
    lora_trainer : LoRATrainer
        Configured trainer instance.

    Returns
    -------
    InferenceAPI
        Loaded inference API ready for classification.

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
        prompt_config=lora_trainer.prompt_config,
    )
    api.load_model()
    return api
