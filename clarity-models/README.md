# Clarity Models

> This module provides a configurable framework for training and serving classification models.
> Supports both transformer encoders and large language models with LoRA fine-tuning, all exposed through a FastAPI
> service.

---

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Format](#data-format)

---

## Overview

### Features

- **Multiple model types**: Encoder (BERT-like), LoRA (OPT/GPT-like), and Classic loaders
- **Configuration-driven**: All aspects controlled via `models.yaml`
- **Multi-model serving**: Serve multiple models in parallel under different endpoints
- **Flexible data processing**: Customizable field names, label mappings, sample sizes
- **TensorBoard integration**: Automatic startup and process management
- **Device detection**: Automatic choice between CUDA, MPS (Apple Silicon), or CPU
- **API**: Models are served via FastAPI with REST endpoints
- **Command Line Interface**: For training and evaluation

### Project Structure

``` yaml
clarity-models/
├── Dockerfile
├── docker-compose.yaml
├── logging.yaml
├── models-training.ipynb   # Jupyter notebook for model training experiments on Google Colab
├── models.yaml
├── app.py                  # FastAPI app loading models from models.yaml
├── models/
│   ├── encoder.py          # Encoder training & inference
│   ├── lora.py             # LoRA training & inference
│   ├── tensorboard_manager.py
│   └── config/             # Config classes for each model type
├── utils/
│   ├── general_utils.py
│   └── logger.py
└── requirements.txt
```

---

## Configuration

All models are defined in [`models.yaml`](models.yaml).

Supported types:

- `classic`: Custom loader function
- `encoder`: Transformer encoder fine-tuning
- `lora`: LLMs with LoRA adapters

### Examples

#### Encoder model

```yaml
- name: "roberta-large"
  type: "encoder"
  enabled: true
  route: "/classify/roberta-large"

  model_config:
    model_name: "roberta-large"
    num_labels: 3

  training_config:
    max_length: 256
    batch_size: 8
    learning_rate: 1e-5
    num_epochs: 5
    eval_strategy: "epoch"
    save_strategy: "epoch"
    early_stopping_patience: 2

  label_config:
    labels:
      - "Clear Reply"
      - "Clear Non-Reply"
      - "Ambivalent"
```

#### LoRA model

```yaml
- name: "opt-1.3b"
  type: "lora"
  enabled: false
  route: "/classify/opt-1-3b"

  model_config:
    model_name: "facebook/opt-1.3b"
    use_8bit: true

  training_config:
    batch_size: 2
    gradient_accumulation_steps: 8
    learning_rate: 3e-4
    num_epochs: 5

  data_config:
    train_sample_size: 600
    valid_sample_size: 200

  tensorboard_config:
    auto_start: true
    port: 6006
```

---

## Usage

### Prerequisites

* [Docker](https://www.docker.com/get-started/) (for containerized execution)
* [Python 3.8+](https://www.python.org/downloads/) (for native execution)

### Build and Run (Docker)

From inside the `clarity-models/` directory:

```bash
# Build and start the container
docker compose up -d
```

```bash
# Rebuild from scratch if needed
docker compose build --no-cache
```

### Native

For training the models it is recommended to run natively with GPU support.

```bash
python3 -m venv venv # Create virtual environment
source venv/bin/activate # Activate virtual environment
pip install -r requirements.txt # Install dependencies
uvicorn app:app # Start uvicorn server
```

### Command Line Interface

In addition to serving models via FastAPI, you can now run training and inference directly from the command line.

```bash
# List available models from models.yaml
python app.py list
# Train a specific model with optional custom config
python app.py train --config custom-config.yaml --model roberta-base train
# Run inference on a QA pair
python app.py test --question "Question?" --context "Context."
```

This is useful for quick experiments or running jobs in environments where an API server is not needed.

### Google Colab / Jupyter Support

A Jupyter notebook is included for interactive training and evaluation, optimized for Google Colab.

File: [`models-training.ipynb`](models-training.ipynb)

### Accessing the FastAPI Service

Exposed ports:

* `8000`: FastAPI service
* `6006`: TensorBoard (if enabled)

Models defined in [`models.yaml`](models.yaml) are exposed via REST. Example:

```bash
curl -X POST "http://localhost:8000/classify/opt-1-3b" \
  -H "Content-Type: application/json" \
  -d '{ "question": "What is the current state of the world?", "context": "Mr. President, what is the current state of the world? - The world is facing numerous challenges including climate change, pandemics, and geopolitical tensions." }'
```

Response:

```json
{
  "clarity_label": "Clear Reply",
  "confidence": 0.89,
  "scores": {
    "Clear Reply": 0.89,
    "Clear Non-Reply": 0.08,
    "Ambivalent": 0.03
  }
}
```

### Logging

Logging configured via [`logging.yaml`](logging.yaml). Default format:

```
2025-10-26 12:00:00 | INFO     | Training started
```

---

## Data Format

Default QA-pair structure:

```json
[
  {
    "question": "Will you invite them to the White House?",
    "context": "Mr. President, I have a question regarding the recent events. Will you invite them to the White House? - Yes, I will.",
    "clarity_label": "Clear Reply"
  }
]
```

Custom field names can be set in `data_config`:

```yaml
data_config:
  label_field: "sentiment"
  question_field: "text"
  context_field: "context"
```