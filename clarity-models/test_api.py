"""
Test runner for the classification API.

This module provides a small utility to load labeled test samples from a JSON
file and send them concurrently to a classification HTTP API. It collects
predictions and computes common classification metrics.

Functions
---------
classify_sample(question, context)
    Send a single classification request and return the predicted label name.
process_entry(entry, index)
    Helper used by worker threads to classify one JSON entry and return labels.
main()
    Load test data, run concurrent requests and print accuracy & macro F1.

Notes
-----
- Retries transient network errors up to MAX_RETRIES with a fixed RETRY_DELAY.
- Uses ThreadPoolExecutor to parallelize requests.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from sklearn.metrics import accuracy_score, f1_score

INPUT_FILE = "../data/cleaned/test.json"
API_URL = "http://localhost:8000/classify/roberta-large"
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_WORKERS = 8


def classify_sample(question, context):
    """Classify a single sample via HTTP API.

    Sends a JSON payload with keys "question" and "context" to the global
    API_URL and returns the predicted class name extracted from the JSON
    response.

    Parameters
    ----------
    question : str
        Question text to classify.
    context : str
        Context text associated with the question.

    Returns
    -------
    str or None
        The predicted label name as returned under the response's "name" key.
        Returns None if the response has no "name" key.

    Raises
    ------
    requests.RequestException
        If the HTTP request fails after all retries.

    Notes
    -----
    - Retries up to MAX_RETRIES with RETRY_DELAY seconds between attempts.
    - A timeout of 30s is applied to each request.
    """
    payload = {"question": question, "context": context}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("name")
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise e
    return None


def process_entry(entry, index):
    """Process a single JSON entry and return (true_label, pred_label).

    This function extracts the required fields from the entry dict,
    calls classify_sample and returns the ground-truth label together with the
    predicted label. Exceptions are caught and logged; on failure (None, None)
    is returned so the caller can skip the sample.

    Parameters
    ----------
    entry : dict
        Dictionary with keys 'question', 'context' and 'clarity_label'.
    index : int
        Sequential index used for logging.

    Returns
    -------
    tuple of (str or None, str or None)
        (true_label, pred_label). Each element is None if unavailable or the
        prediction failed.

    Notes
    -----
    - Safe to call concurrently from multiple threads.
    """
    question = entry.get("question")
    context = entry.get("context")
    true_label = entry.get("clarity_label")
    try:
        pred_label = classify_sample(question, context)
        print(f"{index}: true={true_label}, pred={pred_label}")
        return true_label, pred_label
    except Exception as e:
        print(f"[Fehler bei Eintrag {index}] {e}")
        return None, None


def main():
    """Run the evaluation pipeline and print simple metrics.

    Workflow
    --------
    1. Load JSON test data from INPUT_FILE (expects a list of dicts).
    2. Use a ThreadPoolExecutor to issue classification requests concurrently.
    3. Aggregate valid true/pred pairs and compute accuracy and macro F1-score.
    4. Print results to stdout.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If INPUT_FILE does not exist or cannot be opened.
    ValueError
        If the JSON file cannot be parsed into the expected structure.
    """
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    y_true, y_pred = [], []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_entry, entry, i): i for i, entry in enumerate(data, start=1)}

        for future in as_completed(futures):
            true_label, pred_label = future.result()
            if true_label is not None and pred_label is not None:
                y_true.append(true_label)
                y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")


if __name__ == "__main__":
    main()
