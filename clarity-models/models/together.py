"""Together model inference framework.

This module provides prompt formatting, label extraction utilities and a small
inference wrapper around the Together/OpenAI-style client used in the framework.
"""
import os
import time
from typing import Dict, List

from dotenv import load_dotenv
from dto.dto import (
    ClassificationRequest,
)
from models.config.together_config import (
    TogetherConfig
)
from openai import OpenAI
from utils.logger import logger


# =======================================================================================
# Prompt Formatting Function
# =======================================================================================

def format_prompt(prompt: str, request: ClassificationRequest) -> str:
    """Format a prompt template using fields from a classification request.

    Parameters
    ----------
    prompt : str
        Template string containing placeholders for "question" and "context".
    request : ClassificationRequest
        DTO containing `question` and `context` attributes used to fill the template.

    Returns
    -------
    str
        The formatted prompt ready to be sent to the model.
    """
    prompt = prompt.format(
        question=request.question,
        context=request.context
    )

    return prompt


# =======================================================================================
# Extract label function
# =======================================================================================

def extract_label(valid_labels: List[str], text: str) -> str:
    """Extract a label from free-form model output.

    The extraction is heuristic and tries several strategies in order:
    1. Look for a line containing "Label:" and take the suffix.
    2. Case-insensitive substring match of each valid label.
    3. Check that all words of a label appear in the output.
    4. Match on the first word of the output to a label.
    5. Fall back to the first valid label and emit a warning.

    Parameters
    ----------
    valid_labels : List[str]
        List of valid label strings the function may return.
    text : str
        Free-form text produced by the model.

    Returns
    -------
    str
        One of the entries from ``valid_labels``. If no confident match is found,
        the first label in ``valid_labels`` is returned.
    """
    # Preprocess text to find line with "Label:"
    for line in text.splitlines():
        if "Label:" in line:
            text = line.split("Label:")[-1]
            break

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
# Inference API
# =======================================================================================

class InferenceAPI:
    """Wrapper providing a simple classify interface for a Together/OpenAI client.

    Parameters
    ----------
    together_config : TogetherConfig
        Configuration object containing prompts, labels and runtime options.
    client : OpenAI
        An initialized OpenAI-compatible client instance used for chat completions.

    Attributes
    ----------
    together_config : TogetherConfig
        Stored configuration.
    client : OpenAI
        Stored client instance.
    """

    def __init__(
            self,
            together_config: TogetherConfig = TogetherConfig(),
            client: OpenAI = None
    ):
        self.together_config = together_config
        self.client = client

    def classify(self, data: ClassificationRequest) -> Dict:
        """Classify a single request by calling the model and extracting a label.

        The method will attempt ``together_config.max_retries`` times in case of
        transient errors. The response is expected to contain the label text
        which is then normalised by ``extract_label``.

        Parameters
        ----------
        data : ClassificationRequest
            DTO containing the question and context to classify.

        Returns
        -------
        Dict
            A dictionary containing the selected label under the key ``"name"``.
            If classification fails after retries the first label in the config
            is returned as a fallback.

        Raises
        ------
        ValueError
            If the client was not initialized.
        """
        if self.client is None:
            raise ValueError("Client not initialized for InferenceAPI.")

        start_time = time.time()

        for attempt in range(self.together_config.max_retries):
            try:
                prompt = format_prompt(self.together_config.prompt, data)
                logger.info(f"Classifying {prompt}")

                response = self.client.chat.completions.create(
                    model=self.together_config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.together_config.temperature,
                    max_tokens=self.together_config.max_tokens
                )

                content = response.choices[0].message.content
                label = extract_label(self.together_config.labels, content)

                return {"name": label}

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed after {time.time() - start_time:.3f}s: {e}")
                if attempt < self.together_config.max_retries - 1:
                    time.sleep(self.together_config.retry_delay)

        return {"name": self.together_config.labels[0]}


# =======================================================================================
# Loader Function for API Integration
# =======================================================================================

def load_model(together_config: TogetherConfig) -> InferenceAPI:
    """Load environment variables and create an InferenceAPI instance.

    The function searches for environment files listed in ``together_config.env_files``
    and loads the first one that exists. It then reads the TOGETHER_API_KEY and
    instantiates an OpenAI-compatible client pointing to the Together API base.

    Parameters
    ----------
    together_config : TogetherConfig
        Configuration object that defines env_files, model_name and other options.

    Returns
    -------
    InferenceAPI
        An initialized inference wrapper ready to call ``classify``.
    """
    for env_file in together_config.env_files:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
            break

    together_api_key = os.getenv("TOGETHER_API_KEY")

    if not together_api_key:
        raise ValueError("TOGETHER_API_KEY not found in environment variables.")

    logger.info(f"Using the following model: {together_config.model_name}")
    logger.info(f"Mode: {together_config.mode}")

    client = OpenAI(
        api_key=together_api_key,
        base_url="https://api.together.xyz/v1",
        max_retries=0
    )

    api = InferenceAPI(
        together_config=together_config,
        client=client
    )

    return api
