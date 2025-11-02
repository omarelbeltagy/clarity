"""Configuration and prompt templates for Together model integration.

This module provides default prompt templates and the TogetherConfig dataclass
which centralises model, prompt and runtime options used by the inference code.
"""

from dataclasses import dataclass
from typing import Dict, List

from utils.general_utils import (
    as_int,
    as_float,
    as_bool,
    as_str
)

FINE_TUNE_PROMPT = """
Based on a part of the interview where the interviewer asks a set of questions, classify the type of answer the interviewee provided for the following question.
Respond with exactly one of the following labels: [Clear Reply], [Clear Non-Reply], [Ambivalent].

### Part of the interview ###
{context}

### Question ###
{question}
"""

ZERO_SHOT_PROMPT = """
You are an expert in political sciences and interview analysis.
Your task is to classify the type of responses given by interviewees based on the questions posed by the interviewer.

Based on a segment of the interview in which the interviewer poses a series of questions, classify the type of response provided by the interviewee for the following question using the following taxonomy and then provide a chain of thought explanation for your decision:

1. Clear Reply - The information requested is explicitly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification or declining to answer
3. Ambivalent - The information requested is given in an incomplete way e.g. the answer is too general, partial, implicit, dodging or deflection

---
Here is the segment of the interview that you should analyze:

### Part of the Interview ###
{context}
### Question ###
{question}
---

Return only the label in the format "Label: <label>". No additional text or metadata.
"""

FEW_SHOT_PROMPT = """
You are an expert in political sciences and interview analysis.
Your task is to classify the type of responses given by interviewees based on the questions posed by the interviewer.

Based on a segment of the interview in which the interviewer poses a series of questions, classify the type of response provided by the interviewee for the following question using the following taxonomy and then provide a chain of thought explanation for your decision:

1. Clear Reply - The information requested is explicitly stated (in the requested form)
2. Clear Non-Reply - The information requested is not given at all due to ignorance, need for clarification or declining to answer
3. Ambivalent - The information requested is given in an incomplete way e.g. the answer is too general, partial, implicit, dodging or deflection

---

Here is one small example for each term of the taxonomy:

Question: Do you have your own views about PR at Westminster don’t you?
Answer: I do.
Label: Clear Reply
Explanation: The answer directly gives the info requested.

Question: Are you going to watch television?
Answer: What else is there to do?
Label: Ambivalent
Explanation: They suggest planning to watch TV, despite not explicitly stating it.

Question: Do you like my new dress?
Answer: We are late.
Label: Ambivalent
Explanation: Does not even acknowledge the question and goes straight to another topic.

Question: Did you enjoy the film?
Answer: The directing was great.
Label: Ambivalent
Explanation: Directing is only part of what constitutes a film.

Question: What’s your favorite film?
Answer: Fight Club, Filth, and Hereditary.
Label: Ambivalent
Explanation: The reply gives three movies instead of one, which makes the desired information unclear.

Question: The hypothesis I was discussing, wouldn’t you regard that as a defeat?
Answer: I am not going to prophesy what will happen.
Label: Clear Non-Reply
Explanation: Directly stating they won’t answer.

---
Here is the segment of the interview that you should analyze:

### Part of the Interview ###
{context}
### Question ###
{question}

---

Return only the label in the format "Label: <label>". No additional text or metadata.
"""


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
