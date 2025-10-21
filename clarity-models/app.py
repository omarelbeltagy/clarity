import importlib
import sys
import yaml
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

app = FastAPI()


class QAInput(BaseModel):
    question: str
    answer: str


# Load YAML models config
with open("models.yaml", "r") as f:
    config = yaml.safe_load(f)

loaded_models = {}

# Print a list of all available models from the config
logger.info("Available models in configuration:")
for m in config.get("models", []):
    default_route = f"/classify/{m['name']}"
    route = m.get("route", default_route)

    logger.info(
        f" - {m['name']} (module: {m['module']}, loader: {m['loader']}, route: {route})"
    )

# Dynamically import and load models
for m in config.get("models", []):
    if not m.get("enabled", True):
        logger.debug(f"Skipping disabled model '{m['name']}'")
        continue

    try:
        module = importlib.import_module(m["module"])
        loader = getattr(module, m["loader"])
        logger.info(f"Loading model '{m['name']}'")
        loaded_models[m["name"]] = loader()
        logger.info(f"Loaded model '{m['name']}' from {m['module']}.{m['loader']}")
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
            return api.classify(data.question, data.answer)

        return endpoint


    app.post(route)(make_endpoint(api))
    logger.info(f"Endpoint available at {route}")
