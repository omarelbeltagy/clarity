import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from models.config.encoder_config import (
    EncoderModelConfig,
    EncoderTrainingConfig,
    EncoderDataConfig,
    LabelConfig as EncoderLabelConfig,
)
from models.config.lora_config import (
    LoRAConfig,
    LoRATrainingConfig,
    LoRADataConfig,
    LoRAModelConfig,
    LabelConfig as LoRALabelConfig,
)
from models.config.tensorboard_config import TensorboardConfig
from models.encoder import (
    EncoderTrainer,
    load_model as load_encoder_model
)
from models.lora import (
    LoRATrainer,
    load_model as load_lora_model,
    create_default_format_function,
    PromptConfig
)
from utils.general_utils import (
    get_execution_environment,
    is_running_in_docker
)
from utils.logger import logger

# Global config path
CONFIG_PATH = "models.yaml"

# FastAPI app instance
app = FastAPI()


class QAInput(BaseModel):
    question: str
    answer: str


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    path = config_path or CONFIG_PATH
    logger.info(f"Loading configuration from: {path}")

    if not Path(path).exists():
        logger.error(f"Configuration file not found: {path}")
        sys.exit(1)

    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_lora_model_from_config(model_def: dict):
    """Load a LoRA model from YAML configuration."""
    logger.info(f"Loading LoRA model '{model_def['name']}'")

    # Load configs
    data_config = LoRADataConfig.from_dict(model_def.get("data_config", {}))
    label_config = LoRALabelConfig.from_dict(model_def.get("label_config", {}))

    # Create default format function with dynamic fields
    from models.lora import create_default_format_function
    format_fn = create_default_format_function(data_config)

    # Create prompt config with dynamic format function
    from models.lora import PromptConfig
    prompt_config = PromptConfig(format_function=format_fn)

    # Create trainer
    trainer = LoRATrainer(
        model_config=LoRAModelConfig.from_dict(model_def.get("model_config", {})),
        lora_config=LoRAConfig.from_dict(model_def.get("lora_config", {})),
        training_config=LoRATrainingConfig.from_dict(model_def.get("training_config", {})),
        data_config=LoRADataConfig.from_dict(model_def.get("data_config", {})),
        label_config=label_config,
        prompt_config=prompt_config,
        tensorboard_config=TensorboardConfig.from_dict(model_def.get("tensorboard_config", {})),
    )

    # Load model and train it if needed
    lora_api = load_lora_model(trainer)

    return lora_api


def load_encoder_model_from_config(model_def: dict):
    """Load an encoder model from YAML configuration."""
    logger.info(f"Loading encoder model '{model_def['name']}'")

    # Create trainer
    trainer = EncoderTrainer(
        model_config=EncoderModelConfig.from_dict(model_def.get("model_config", {})),
        training_config=EncoderTrainingConfig.from_dict(model_def.get("training_config", {})),
        data_config=EncoderDataConfig.from_dict(model_def.get("data_config", {})),
        label_config=EncoderLabelConfig.from_dict(model_def.get("label_config", {})),
        tensorboard_config=TensorboardConfig.from_dict(model_def.get("tensorboard_config", {})),
    )

    # Load model and train it if needed
    encoder_api = load_encoder_model(trainer)

    return encoder_api


def load_classic_model_from_config(model_def: dict):
    """Load a classic model using module/loader pattern."""
    logger.info(f"Loading classic model '{model_def['name']}'")

    module = importlib.import_module(model_def["module"])
    loader = getattr(module, model_def["loader"])
    return loader()


def get_model_by_name(config: dict, model_name: Optional[str] = None):
    """Get model definition by name or return first enabled model."""
    models = config.get("models", [])

    if not models:
        logger.error("No models found in configuration")
        sys.exit(1)

    if model_name:
        # Find specific model
        for m in models:
            if m.get("name") == model_name:
                return m
        logger.error(f"Model '{model_name}' not found in configuration")
        sys.exit(1)
    else:
        # Return first enabled model
        for m in models:
            if m.get("enabled", True):
                logger.info(f"Using first enabled model: {m.get('name')}")
                return m
        logger.error("No enabled models found in configuration")
        sys.exit(1)


# =======================================================================================
# CLI Commands
# =======================================================================================

def cmd_train(args):
    """Train a model from configuration."""
    logger.info("=" * 60)
    logger.info("TRAINING MODE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(args.config)
    model_def = get_model_by_name(config, args.model)

    logger.info(f"Model: {model_def.get('name')}")
    logger.info(f"Type: {model_def.get('type')}")
    logger.info(f"Environment: {get_execution_environment()}")
    logger.info("=" * 60)

    model_type = model_def.get("type", "classic")

    try:
        if model_type == "encoder":
            # Train encoder model
            data_config = EncoderDataConfig.from_dict(model_def.get("data_config", {}))
            label_config = EncoderLabelConfig.from_dict(model_def.get("label_config", {}))

            # Enable TensorBoard if requested
            tb_config = TensorboardConfig.from_dict(model_def.get("tensorboard_config", {}))
            if args.tensorboard:
                tb_config.auto_start = True

            trainer = EncoderTrainer(
                model_config=EncoderModelConfig.from_dict(model_def.get("model_config", {})),
                training_config=EncoderTrainingConfig.from_dict(model_def.get("training_config", {})),
                data_config=data_config,
                label_config=label_config,
                tensorboard_config=tb_config,
            )

            # Train
            trainer.train()
            logger.info("✓ Training completed successfully!")

        elif model_type == "lora":
            # Train LoRA model
            data_config = LoRADataConfig.from_dict(model_def.get("data_config", {}))
            label_config = LoRALabelConfig.from_dict(model_def.get("label_config", {}))
            format_fn = create_default_format_function(data_config)

            # Enable TensorBoard if requested
            tb_config = TensorboardConfig.from_dict(model_def.get("tensorboard_config", {}))
            if args.tensorboard:
                tb_config.auto_start = True

            trainer = LoRATrainer(
                model_config=LoRAModelConfig.from_dict(model_def.get("model_config", {})),
                lora_config=LoRAConfig.from_dict(model_def.get("lora_config", {})),
                training_config=LoRATrainingConfig.from_dict(model_def.get("training_config", {})),
                data_config=data_config,
                label_config=label_config,
                prompt_config=PromptConfig(format_function=format_fn),
                tensorboard_config=tb_config,
            )

            # Train
            trainer.train()
            logger.info("✓ Training completed successfully!")

        else:
            logger.error(f"Training not supported for model type: {model_type}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_test(args):
    """Test a trained model."""
    logger.info("=" * 60)
    logger.info("TESTING MODE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(args.config)
    model_def = get_model_by_name(config, args.model)

    logger.info(f"Model: {model_def.get('name')}")
    logger.info(f"Type: {model_def.get('type')}")
    logger.info("=" * 60)

    model_type = model_def.get("type", "classic")

    try:
        # Load the model
        if model_type == "encoder":
            api = load_encoder_model_from_config(model_def)
        elif model_type == "lora":
            api = load_lora_model_from_config(model_def)
        elif model_type == "classic":
            api = load_classic_model_from_config(model_def)
        else:
            logger.error(f"Unknown model type: {model_type}")
            sys.exit(1)

        # Test with provided input or interactive mode
        if args.question and args.answer:
            # Single prediction
            logger.info("\nInput:")
            logger.info(f"  Question: {args.question}")
            logger.info(f"  Answer: {args.answer}")

            result = api.classify(
                question=args.question,
                answer=args.answer
            )

            logger.info("\n" + "=" * 60)
            logger.info("PREDICTION RESULT")
            logger.info("=" * 60)

            if "clarity_label" in result:
                logger.info(f"Label: {result['clarity_label']}")
                logger.info(f"Confidence: {result['confidence']:.2%}")
                logger.info("\nAll Scores:")
                for label, score in result['scores'].items():
                    logger.info(f"  {label}: {score:.2%}")
            elif "extracted_result" in result:
                logger.info(f"Result: {result['extracted_result']}")
                logger.info(f"Generated: {result['generated_text']}")
            else:
                logger.info(f"Result: {result}")

            logger.info("=" * 60)

        else:
            # Interactive mode
            logger.info("\nInteractive Testing Mode")
            logger.info("Enter 'quit' or 'exit' to stop\n")

            while True:
                try:
                    question = input("Question: ").strip()
                    if question.lower() in ['quit', 'exit', 'q']:
                        break

                    answer = input("Answer: ").strip()
                    if answer.lower() in ['quit', 'exit', 'q']:
                        break

                    if not question or not answer:
                        logger.warning("Both question and answer are required")
                        continue

                    result = api.classify(question=question, answer=answer)

                    print("\n" + "-" * 60)
                    if "clarity_label" in result:
                        print(f"Label: {result['clarity_label']}")
                        print(f"Confidence: {result['confidence']:.2%}")
                    elif "extracted_result" in result:
                        print(f"Result: {result['extracted_result']}")
                    print("-" * 60 + "\n")

                except KeyboardInterrupt:
                    print("\n")
                    break
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_list(args):
    """List all models in configuration."""
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("AVAILABLE MODELS")
    logger.info("=" * 60)

    for m in config.get("models", []):
        status = "✓ enabled" if m.get("enabled", True) else "✗ disabled"
        logger.info(f"{m['name']:<20} | {m.get('type', 'classic'):<10} | {status}")

    logger.info("=" * 60)


# =======================================================================================
# FastAPI Server Mode
# =======================================================================================

def initialize_api_server():
    """Initialize FastAPI server with models."""
    logger.info("Starting Clarity Models API...")
    logger.info(f"Running in Docker: {is_running_in_docker()}")
    logger.info(f"Environment: {get_execution_environment()}")

    config = load_config()
    loaded_models = {}

    # Print a list of all available models from the config
    logger.info("Available models in configuration:")
    for m in config.get("models", []):
        model_type = m.get("type", "classic")
        default_route = f"/classify/{m['name']}"
        route = m.get("route", default_route)
        logger.info(f" - {m['name']} (type: {model_type}, route: {route})")

    # Dynamically import and load models
    for m in config.get("models", []):
        if not m.get("enabled", True):
            logger.debug(f"Skipping disabled model '{m['name']}'")
            continue

        try:
            model_type = m.get("type", "classic")

            if model_type == "lora":
                loaded_models[m["name"]] = load_lora_model_from_config(m)
            elif model_type == "encoder":
                loaded_models[m["name"]] = load_encoder_model_from_config(m)
            elif model_type == "classic":
                if "module" not in m or "loader" not in m:
                    logger.error(f"Classic model '{m['name']}' requires 'module' and 'loader' fields")
                    continue
                loaded_models[m["name"]] = load_classic_model_from_config(m)
            else:
                logger.error(f"Unknown model type '{model_type}' for model '{m['name']}'")
                continue

            logger.info(f"Loaded model '{m['name']}' successfully")

        except Exception as e:
            logger.error(f"Failed to load model '{m['name']}': {e}")

    # Dynamically create endpoints
    for name, api in loaded_models.items():
        route = next(
            (m["route"] for m in config["models"] if m["name"] == name),
            f"/classify/{name}",
        )

        def make_endpoint(api):
            async def endpoint(data: QAInput):
                return api.classify(
                    question=data.question,
                    answer=data.answer
                )

            return endpoint

        app.post(route)(make_endpoint(api))
        logger.info(f"Endpoint available at {route}")

    # Store loaded models in app state
    app.state.loaded_models = loaded_models
    app.state.config = config


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Clarity Models API",
        "models": list(app.state.loaded_models.keys()),
        "endpoints": [
            next((m["route"] for m in app.state.config["models"] if m["name"] == name),
                 f"/classify/{name}")
            for name in app.state.loaded_models.keys()
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": len(app.state.loaded_models)}


# =======================================================================================
# Main Entry Point
# =======================================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Clarity Models - Training and Inference Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python app.py list
  
  # Train a model
  python app.py train
  python app.py train --model roberta-base --tensorboard
  python app.py train --config my_models.yaml
  
  # Test a model (interactive)
  python app.py test
  
  # Test a model (single prediction)
  python app.py test --question "Question?" --answer "Answer."
  python app.py test --model opt-1.3b --question "Question?" --answer "Answer."
  
  # Run as API server
  uvicorn app:app --host 0.0.0.0 --port 8000
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="models.yaml",
        help="Path to models configuration file (default: models.yaml)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    subparsers.add_parser("list", help="List all available models")

    # Train command
    parser_train = subparsers.add_parser("train", help="Train a model")
    parser_train.add_argument(
        "--model",
        type=str,
        help="Model name to train (default: first enabled model)"
    )
    parser_train.add_argument(
        "--tensorboard",
        action="store_true",
        help="Start TensorBoard during training"
    )

    # Test command
    parser_test = subparsers.add_parser("test", help="Test a trained model")
    parser_test.add_argument(
        "--model",
        type=str,
        help="Model name to test (default: first enabled model)"
    )
    parser_test.add_argument(
        "--question",
        type=str,
        help="Question text for prediction"
    )
    parser_test.add_argument(
        "--answer",
        type=str,
        help="Answer text for prediction"
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    # CLI mode
    main()
else:
    # API server mode
    initialize_api_server()
