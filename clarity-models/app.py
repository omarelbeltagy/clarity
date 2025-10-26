import importlib
import sys
import yaml
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from models.lora import (
    LoRATrainer,
    LoRAConfig,
    TrainingConfig,
    DataConfig,
    ModelConfig,
    TensorboardConfig,
    load_model as load_lora_model
)
from utils.general_utils import (
    get_execution_environment,
    is_running_in_docker
)
from utils.logger import logger

logger.info("Starting Clarity Models API...")
logger.info(f"Running in Docker: {is_running_in_docker()}")
logger.info(f"Environment: {get_execution_environment()}")

app = FastAPI()


class QAInput(BaseModel):
    question: str
    answer: str


# Load YAML models config
with open("models.yaml", "r") as f:
    config = yaml.safe_load(f)

loaded_models = {}


def load_lora_model_from_config(model_def: dict):
    """Load a LoRA model from YAML configuration."""
    logger.info(f"Loading LoRA model '{model_def['name']}'")

    # Create trainer
    trainer = LoRATrainer(
        model_config=ModelConfig.from_dict(model_def.get("model_config", {})),
        lora_config=LoRAConfig.from_dict(model_def.get("lora_config", {})),
        training_config=TrainingConfig.from_dict(model_def.get("training_config", {})),
        data_config=DataConfig.from_dict(model_def.get("data_config", {})),
        tensorboard_config=TensorboardConfig.from_dict(model_def.get("tensorboard_config", {})),
    )

    # Load model and train it if needed
    lora_api = load_lora_model(trainer)

    return lora_api


def load_classic_model_from_config(model_def: dict):
    """Load a classic model using module/loader pattern."""
    logger.info(f"Loading classic model '{model_def['name']}'")

    module = importlib.import_module(model_def["module"])
    loader = getattr(module, model_def["loader"])
    return loader()


# Print a list of all available models from the config
logger.info("Available models in configuration:")
for m in config.get("models", []):
    model_type = m.get("type", "classic")
    default_route = f"/classify/{m['name']}"
    route = m.get("route", default_route)

    logger.info(
        f" - {m['name']} (type: {model_type}, route: {route})"
    )

# Dynamically import and load models
for m in config.get("models", []):
    if not m.get("enabled", True):
        logger.debug(f"Skipping disabled model '{m['name']}'")
        continue

    try:
        model_type = m.get("type", "classic")

        if model_type == "lora":
            # Load LoRA model
            loaded_models[m["name"]] = load_lora_model_from_config(m)
        elif model_type == "classic":
            # Load classic model
            if "module" not in m or "loader" not in m:
                logger.error(
                    f"Classic model '{m['name']}' requires 'module' and 'loader' fields"
                )
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
