"""Main application entrypoint for Clarity Models.

Provides:
- CLI subcommands to list, train and test models described in a YAML config.
- FastAPI application initialization to expose classification endpoints.

This module supports two usage modes:
- CLI mode (train/test/list): executed when run as __main__.
- API server mode: import this module and serve `app` with an ASGI server.

Key functions
-------------
load_config
    Load YAML configuration from disk.
load_*_model_from_config
    Construct trainer/loader objects for different model types.
initialize_api_server
    Load configured models and register endpoints dynamically.
"""

import argparse
import importlib
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score
from torch.multiprocessing import freeze_support
from typing import Optional

from dto.dto import (
    ClassificationRequest,
)
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
    PromptConfig
)
from models.config.tensorboard_config import TensorboardConfig
from models.config.together_config import TogetherConfig
from models.encoder import (
    EncoderTrainer,
    load_model as load_encoder_model
)
from models.lora import (
    LoRATrainer,
    load_model as load_lora_model,
)
from models.together import (
    load_model as load_together_model,
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


def load_config(config_path: str = None) -> dict:
    """Load models configuration from a YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration. If None, the module-level CONFIG_PATH is used.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    SystemExit
        Exits the process if the configuration file is missing or unreadable.

    Examples
    --------
    >>> config = load_config("my_models.yaml")
    """
    path = config_path or CONFIG_PATH
    logger.info(f"Loading configuration from: {path}")

    if not Path(path).exists():
        logger.error(f"Configuration file not found: {path}")
        sys.exit(1)

    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_lora_model_from_config(model_def: dict):
    """Instantiate and return a LoRA model API from a configuration dict.

    The function builds a LoRATrainer from the provided model_def and then
    calls the lower-level loader to return an inference API object.

    Parameters
    ----------
    model_def : dict
        Model definition as found in the YAML config; should contain
        'model_config', 'lora_config', 'training_config', 'data_config', etc.

    Returns
    -------
    object
        Model API object (expected to implement classify(data) for inference).

    Raises
    ------
    Exception
        Propagates exceptions raised during trainer construction or model loading.
    """
    logger.info(f"Loading LoRA model '{model_def['name']}'")

    # Create trainer
    trainer = LoRATrainer(
        model_config=LoRAModelConfig.from_dict(model_def.get("model_config", {})),
        lora_config=LoRAConfig.from_dict(model_def.get("lora_config", {})),
        training_config=LoRATrainingConfig.from_dict(model_def.get("training_config", {})),
        data_config=LoRADataConfig.from_dict(model_def.get("data_config", {})),
        label_config=LoRALabelConfig.from_dict(model_def.get("label_config", {})),
        prompt_config=PromptConfig.from_dict(model_def.get("prompt_config", {})),
        tensorboard_config=TensorboardConfig.from_dict(model_def.get("tensorboard_config", {})),
    )

    # Load model and train it if needed
    lora_api = load_lora_model(trainer)

    return lora_api


def load_encoder_model_from_config(model_def: dict):
    """Instantiate and return an encoder model API from configuration.

    Parameters
    ----------
    model_def : dict
        Encoder model definition from the YAML config.

    Returns
    -------
    object
        Model API object suitable for inference (implements classify).

    Raises
    ------
    Exception
        If instantiation or loading fails.
    """
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


def load_together_model_from_config(model_def: dict):
    logger.info(f"Loading together model '{model_def['name']}'")

    if model_def.get("config"):
        config = TogetherConfig.from_dict(model_def.get("config", {}))
    elif model_def.get("together_config"):
        config = TogetherConfig.from_dict(model_def.get("together_config", {}))
    elif model_def.get("model_config"):
        config = TogetherConfig.from_dict(model_def.get("model_config", {}))
    else:
        config = TogetherConfig()

    together_api = load_together_model(config)

    return together_api


def load_classic_model_from_config(model_def: dict):
    """Dynamically import and invoke a loader for a classic model.

    The expected pattern is to have 'module' and 'loader' keys in the config.
    The loader is looked up in the imported module and called with no args.

    Parameters
    ----------
    model_def : dict
        Classic model config containing 'module' and 'loader' strings.

    Returns
    -------
    object
        The model instance returned by the loader.

    Raises
    ------
    ImportError
        If the specified module cannot be imported.
    AttributeError
        If the loader attribute cannot be found in the imported module.
    """
    logger.info(f"Loading classic model '{model_def['name']}'")

    module = importlib.import_module(model_def["module"])
    loader = getattr(module, model_def["loader"])
    return loader()


def get_model_by_name(config: dict, model_name: Optional[str] = None):
    """Select a model entry by name or return the first enabled model.

    Parameters
    ----------
    config : dict
        Parsed configuration containing a 'models' sequence.
    model_name : str, optional
        Name of the model to select. If omitted, the first enabled model is returned.

    Returns
    -------
    dict
        The selected model definition.

    Raises
    ------
    SystemExit
        If no models exist, the named model is missing, or no enabled models exist.
    """
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
    """CLI handler to train a configured model.

    Constructs the appropriate trainer (encoder or lora) from the model
    configuration and runs trainer.train().

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the 'train' command. Typical attributes:
        - config: str path to config
        - model: optional model name
        - tensorboard: bool to enable TensorBoard auto-start

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        On configuration errors or if training fails fatally.
    """
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

            # Enable TensorBoard if requested
            tb_config = TensorboardConfig.from_dict(model_def.get("tensorboard_config", {}))
            if args.tensorboard:
                tb_config.auto_start = True
            else:
                tb_config.auto_start = False

            trainer = LoRATrainer(
                model_config=LoRAModelConfig.from_dict(model_def.get("model_config", {})),
                lora_config=LoRAConfig.from_dict(model_def.get("lora_config", {})),
                training_config=LoRATrainingConfig.from_dict(model_def.get("training_config", {})),
                data_config=LoRADataConfig.from_dict(model_def.get("data_config", {})),
                label_config=LoRALabelConfig.from_dict(model_def.get("label_config", {})),
                prompt_config=PromptConfig.from_dict(model_def.get("prompt_config", {})),
                tensorboard_config=tb_config,
            )

            # Train
            trainer.train()
            logger.info("Training completed successfully!")

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
    """CLI handler to test a configured model.

    Loads the specified model and performs inference based on provided
    question/context inputs, either interactively or from a json file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the 'test' command.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        On unrecoverable errors during model loading or inference.
    """
    logger.info("=" * 60)
    logger.info("TESTING MODE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(args.config)
    model_def = get_model_by_name(config, args.model)

    logger.info(f"Model: {model_def.get('name')}")
    model_type = model_def.get("type", "classic")
    logger.info(f"Type: {model_type}")
    logger.info("=" * 60)

    try:
        # Load the model
        if model_type == "encoder":
            api = load_encoder_model_from_config(model_def)
        elif model_type == "lora":
            api = load_lora_model_from_config(model_def)
        elif model_type == "classic":
            api = load_classic_model_from_config(model_def)
        elif model_type == "together":
            api = load_together_model_from_config(model_def)
        else:
            logger.error(f"Unknown model type: {model_type}")
            sys.exit(1)

        # Test with provided input or interactive mode
        if args.question and args.context:
            # Single prediction
            logger.info("\nInput:")
            logger.info(f"  Question: {args.question}")
            logger.info(f"  Context: {args.context}")

            classification_request = ClassificationRequest(
                question=args.question,
                context=args.context,
            )

            result = api.classify(data=classification_request)

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

        # Test batch predictions from json file
        elif args.file:
            input_path = Path(args.file)
            if not input_path.exists():
                logger.error(f"Input file not found: {input_path}")
                sys.exit(1)

            logger.info(f"Batch Testing Mode - Input File: {input_path}")
            with open(input_path, "r", encoding="utf-8") as f:
                import json
                test_data = json.load(f)

            def process_entry(entry, index, api_instance):
                entry_question = entry.get("question")
                entry_context = entry.get("context")
                entry_true_label = entry.get("clarity_label", None)

                if not entry_question or not entry_context:
                    logger.warning(f"Entry {index} missing question or context, skipping.")
                    return None, None

                try:
                    request = ClassificationRequest(
                        question=entry_question,
                        context=entry_context,
                    )

                    classification_result = api_instance.classify(data=request)

                    logger.info("-" * 60)
                    logger.info(f"Entry {index}:")

                    entry_pred_label = classification_result['name']

                    return entry_true_label, entry_pred_label
                except Exception as e:
                    logger.error(f"[Error processing entry {index}] {e}")
                    return None, None

            y_true, y_pred = [], []

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_entry, entry, i, api): i for i, entry in
                           enumerate(test_data, start=1)}

                for future in as_completed(futures):
                    true_label, pred_label = future.result()
                    if true_label is not None and pred_label is not None:
                        y_true.append(true_label)
                        y_pred.append(pred_label)
                        logger.info(f"  True Label: {true_label}, Predicted Label: {pred_label}")
                    elif pred_label is not None:
                        logger.info(f"  Predicted Label: {pred_label}")

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            logger.info("=" * 60)
            logger.info("BATCH TESTING RESULTS")
            logger.info("=" * 60)
            logger.info(f"Total Samples: {len(y_true)}")
            logger.info(f"Accuracy: {acc:.4f}")
            logger.info(f"Macro F1: {f1:.4f}")
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

                    context = input("Context: ").strip()
                    if context.lower() in ['quit', 'exit', 'q']:
                        break

                    if not question or not context:
                        logger.warning("Both question and context are required")
                        continue

                    classification_request = ClassificationRequest(
                        question=args.question,
                        context=args.context,
                    )

                    result = api.classify(data=classification_request)

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
    """List models defined in the configuration file.

    Parameters
    ----------
    args : argparse.Namespace
        CLI args (only 'config' is relevant).

    Returns
    -------
    None
    """
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
    """Initialize FastAPI app and dynamically register classification endpoints.

    Loads all enabled models from config, constructs model APIs and registers
    a POST endpoint for each model at its configured route.

    Returns
    -------
    None

    Notes
    -----
    - Loaded model objects are stored in app.state.loaded_models.
    - Endpoints accept a pydantic ClassificationRequest and call api.classify(data).
    """
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
            async def endpoint(data: ClassificationRequest):
                return api.classify(data)

            return endpoint

        app.post(route)(make_endpoint(api))
        logger.info(f"Endpoint available at {route}")

    # Store loaded models in app state
    app.state.loaded_models = loaded_models
    app.state.config = config


@app.get("/")
async def root():
    """Return basic service and model metadata.

    Returns
    -------
    dict
        Service name, list of loaded model names and their endpoint routes.
    """
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
    """Simple health check endpoint.

    Returns
    -------
    dict
        Health status and number of loaded models.
    """
    return {"status": "healthy", "models_loaded": len(app.state.loaded_models)}


# =======================================================================================
# Main Entry Point
# =======================================================================================

def main():
    """Parse CLI arguments and dispatch to subcommands.

    Recognized subcommands: list, train, test. When used as an API server,

    Returns
    -------
    None

    Notes
    -----
    - Supports 'list', 'train' and 'test' subcommands.
    - For API server usage, import this module and run via ASGI server (uvicorn).
    """
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
  python app.py test --question "Question?" --context "Context."
  python app.py test --model opt-1.3b --question "Question?" --context "Context."
  
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
        "--context",
        type=str,
        help="Context text for prediction"
    )
    parser_test.add_argument(
        "--question",
        type=str,
        help="Question text for prediction"
    )
    parser_test.add_argument(
        "--file",
        type=str,
        help="File containing JSON input for batch predictions"
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
    freeze_support()
    main()
else:
    # API server mode
    initialize_api_server()
