"""Logger configuration using loguru and a YAML file.

This module loads logging configuration from 'logging.yaml' and configures
loguru accordingly. Handlers using the string "sys.stdout" are remapped to
the actual sys.stdout object so the YAML file can remain portable.

Notes
-----
- If 'logging.yaml' is missing or malformed the import may raise an error.
"""

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
