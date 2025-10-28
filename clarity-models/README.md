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

### Project Structure

``` yaml
clarity-models/
├── Dockerfile
├── docker-compose.yaml
├── logging.yaml
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

### Native (recommended for training)

For training the models it is recommended to run natively with GPU support.

```bash
python3 -m venv venv # Create virtual environment
source venv/bin/activate # Activate virtual environment
pip install -r requirements.txt # Install dependencies
uvicorn app:app # Start uvicorn server
```

### Accessing the Service

Exposed ports:

* `8000`: FastAPI service
* `6006`: TensorBoard (if enabled)

Models defined in `models.yaml` are exposed via REST. Example:

```bash
curl -X POST "http://localhost:8000/classify/roberta-large" \
  -H "Content-Type: application/json" \
  -d '{ "question": "What is your favorite color?", "answer": "Blue." }'
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
    "answer": "We are ready if they are serious.",
    "clarity_label": "Clear Reply"
  }
]
```

Custom field names can be set in `data_config`:

```yaml
data_config:
  label_field: "sentiment"
  text_field_1: "text"
  text_field_2: "context"
```