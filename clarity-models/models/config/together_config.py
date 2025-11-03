"""Configuration and prompt templates for Together model integration.

This module provides default prompt templates and the TogetherConfig dataclass
which centralises model, prompt and runtime options used by the inference code.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import yaml
from utils.general_utils import (
    SafeDict,
    as_int,
    as_float,
    as_bool,
    as_str
)


def load_default_prompt(prompt_env: str, prompt_default_file: str) -> str:
    """Load a prompt template from a YAML file, injecting taxonomy data.

    Parameters
    ----------
    prompt_env : str
        Environment variable name to override the prompt file path.
    prompt_default_file : str
        Default path to the prompt YAML file.
    Returns
    -------
    str
        The prompt template string with taxonomy data injected.
    Raises
    ------
    FileNotFoundError
        If the prompt file or taxonomy file cannot be found.
    """
    taxonomy_default_path = os.getenv("TAXONOMY_FILE", "../assets/taxonomy/clarity-categories.yaml")
    if not os.path.exists(taxonomy_default_path):
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_default_path}")
    taxonomy_str = build_taxonomy_string(taxonomy_default_path)

    path = os.getenv(prompt_env, prompt_default_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = {"taxonomy": taxonomy_str}
        return yaml.safe_load(f).get("prompt", "").format_map(SafeDict(data))


def build_taxonomy_string(file_path: str) -> str:
    """Build a taxonomy string from a YAML file for inclusion in prompts.

    Parameters
    ----------
    file_path : str
        Path to the YAML file containing the taxonomy definition.
    Returns
    -------
    str
        Formatted taxonomy string with numbered categories.
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    categories = data.get("categories", [])
    result = []
    for i, cat in enumerate(categories, start=1):
        result.append(f"{i}. {cat['name']} - {cat['description'].strip()}")

    return "\n".join(result)


# Default prompt templates loaded from YAML files with taxonomy injection
FINE_TUNE_PROMPT = load_default_prompt("FINE_TUNE_PROMPT_FILE", "../assets/prompts/lora.yaml")
ZERO_SHOT_PROMPT = load_default_prompt("ZERO_SHOT_PROMPT_FILE", "../assets/prompts/zero-shot.yaml")
FEW_SHOT_PROMPT = load_default_prompt("FEW_SHOT_PROMPT_FILE", "../assets/prompts/few-shot.yaml")


@dataclass
class TogetherConfig:
    """Configuration for Together model usage.

    Parameters
    ----------
    model_name : str
        Model identifier used for the API (default provided).
    mode : str
        Prompting mode e.g. "few-shot", "zero-shot", "fine-tune".
    prompt : str
        Prompt template used to query the model.
    env_files : List[str]
        Candidate paths for environment files to load API keys from.
    labels : List[str]
        Allowed labels for classification tasks.
    max_retries : int
        Number of retry attempts on transient API failures.
    max_tokens : int
        Maximum number of tokens to request from the model.
    temperature : float
        Sampling temperature for the model.
    retry_delay : int
        Delay in seconds between retry attempts.
    """

    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    mode: str = "few-shot"
    prompt: str = FEW_SHOT_PROMPT
    env_files: List[str] = None
    labels: List[str] = None
    max_retries: int = 3
    max_tokens: int = 4096
    temperature: float = 0.7
    retry_delay: int = 2

    def __post_init__(self):
        """Set sensible defaults for env_files and labels if not provided.

        Notes
        -----
        This method mutates the instance to ensure that ``env_files`` and
        ``labels`` are always lists even if None was passed during construction.
        """
        if self.env_files is None:
            self.env_files = [
                "/app/data/.env",
                "./.env",
                "../.env",
            ]
        if self.labels is None:
            self.labels = ["Ambivalent", "Clear Reply", "Clear Non-Reply"]

    @classmethod
    def from_dict(cls, cfg: Dict) -> "TogetherConfig":
        """Create a TogetherConfig from a plain dictionary.

        The method reads common configuration keys and applies type conversions
        using utility helpers. Keys supported (case-sensitive): 'env_file',
        'env_files', 'model', 'mode', 'prompt', 'labels', 'max_retries',
        'max_tokens', 'temperature', 'retry_delay'.

        Parameters
        ----------
        cfg : Dict
            Mapping with optional configuration values.

        Returns
        -------
        TogetherConfig
            Configured instance populated from the provided dict.
        """
        instance = cls()
        if "env_file" in cfg:
            instance.env_files = [as_str(cfg["env_file"], "./.env")]
        if "env_files" in cfg:
            instance.train_files = cfg["env_files"]
        if "model_name" in cfg:
            instance.model_name = as_str(cfg["model_name"], "mistralai/Mixtral-8x7B-Instruct-v0.1")
        if "mode" in cfg:
            instance.mode = as_str(cfg["mode"], "few-shot")
        if "prompt" in cfg:
            instance.prompt = as_str(cfg["prompt"], FEW_SHOT_PROMPT)
        else:
            if "mode" in cfg:
                mode = as_str(cfg["mode"], "few-shot")
                if mode == "zero-shot":
                    instance.prompt = as_str(cfg.get("prompt", ZERO_SHOT_PROMPT), ZERO_SHOT_PROMPT)
                elif mode == "few-shot":
                    instance.prompt = as_str(cfg.get("prompt", FEW_SHOT_PROMPT), FEW_SHOT_PROMPT)
                elif mode == "fine-tune":
                    instance.prompt = as_str(cfg.get("prompt", FINE_TUNE_PROMPT), FINE_TUNE_PROMPT)
                else:
                    instance.prompt = as_str(cfg.get("prompt", FEW_SHOT_PROMPT), FEW_SHOT_PROMPT)
        if "labels" in cfg:
            instance.labels = cfg["labels"]
        if "max_retries" in cfg:
            instance.max_retries = as_int(cfg["max_retries"], 3)
        if "max_tokens" in cfg:
            instance.max_tokens = as_int(cfg["max_tokens"], 4096)
        if "temperature" in cfg:
            instance.temperature = as_float(cfg["temperature"], 0.7)
        if "retry_delay" in cfg:
            instance.retry_delay = as_int(cfg["retry_delay"], 2)
        return instance
