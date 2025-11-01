import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from sklearn.metrics import accuracy_score, f1_score

INPUT_FILE = "../data/cleaned/test.json"
API_URL = "http://localhost:8000/classify/roberta-base"
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_WORKERS = 8


def classify_sample(question, context):
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
