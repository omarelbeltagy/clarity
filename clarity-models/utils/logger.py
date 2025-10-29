import sys
import yaml
from loguru import logger

# Load logging configuration from YAML
with open("logging.yaml", "r") as f:
    log_config = yaml.safe_load(f)
    for handler in log_config.get("handlers", []):
        if handler.get("sink") == "sys.stdout":
            handler["sink"] = sys.stdout
logger.configure(**log_config)
