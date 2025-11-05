# Clarity Models

- [📁 Trained Models](https://drive.google.com/drive/folders/1C3q9jZ-92H3tPpaPJaXDgTAHIFmHs1w1?usp=share_link)

> This module provides a configurable framework for training and serving classification models.
> Supports both transformer encoders and large language models with LoRA fine-tuning, all exposed through a FastAPI
> service.

---

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Format](#data-format)
- [Testing](#testing)
- [Fine-Tuning with Together](#fine-tuning-with-together)

---

## Overview

### Features

- **Multiple model types**: Encoder (BERT-like), LoRA (OPT/GPT-like), and Classic loaders
- **Together API support**: Remote/hosted models via Together.ai for few-shot/zero-shot inference and fine-tuned models
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
├── docker-compose.yaml     # Docker Compose setup
├── logging.yaml            # Logging configuration
├── models-training.ipynb   # Jupyter notebook for model training experiments on Google Colab
├── models.yaml             # Model configuration file
├── app.py                  # FastAPI app loading models from models.yaml
├── colab.ipynb             # Google Colab setup notebook
├── dto/                    # Data Transfer Objects for API requests/responses
│   ├── ...
├── models/
│   ├── encoder.py          # Encoder training & inference
│   ├── lora.py             # LoRA training & inference
│   ├── tensorboard_manager.py
│   ├── together.py         # Together API integration
│   └── config/             # Config classes for each model type
│       ├── ...
├── utils/
│   ├── ...
└── requirements.txt        # Python dependencies
```

---

## Configuration

All models are defined in [`models.yaml`](models.yaml).

Supported types:

- `classic`: Custom loader function
- `encoder`: Transformer encoder fine-tuning
- `lora`: LLMs with LoRA adapters
- `together`: Remote/hosted models accessed via the Together API (few-shot/zero-shot/fine-tune). Together Models are
  only available in the `test` mode for inference.

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

#### Together model

```yaml
- name: "Llama-Guard-4-12B"
  type: "together"

  config:
    model_name: "meta-llama/Llama-Guard-4-12B"  # HF / Together / local id
    mode: "few-shot"                            # "few-shot" | "zero-shot" | "fine-tune"
    prompt: null                                # Optional custom prompt template
    env_files: # Candidate .env files to load API keys from
      - "/app/data/.env"
      - "./.env"
    labels:
      - "Clear Reply"
      - "Clear Non-Reply"
      - "Ambivalent"
    max_retries: 3
    max_tokens: 4096
    temperature: 0.7
    retry_delay: 2
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
python app.py train --config custom-config.yaml train --model roberta-base
```

For inference without starting the API server see the [Testing](#testing) section.

### Google Colab / Jupyter Support

A Jupyter notebook is included for interactive training and evaluation, optimized for Google Colab.

File: [colab.ipynb](colab.ipynb)

### Accessing the FastAPI Service

Exposed ports:

* `8000`: FastAPI service
* `6006`: TensorBoard (if enabled)

Models defined in [models.yaml](models.yaml) are exposed via REST. Example:

```bash
curl -X POST "http://localhost:8000/classify/opt-1-3b" \
  -H "Content-Type: application/json" \
  -d '{ "question": "What is the current state of the world?", "context": "Mr. President, what is the current state of the world? - The world is facing numerous challenges including climate change, pandemics, and geopolitical tensions." }'
```

Response:

```json
{
  "name": "Clear Reply",
  "confidence": 0.89,
  "scores": {
    "Clear Reply": 0.89,
    "Clear Non-Reply": 0.08,
    "Ambivalent": 0.03
  }
}
```

### Logging

Logging configured via [logging.yaml](logging.yaml). Default format:

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

---

## Testing

To test a Models performance, you can use the `test` argument with the [app.py](app.py) script.

### Single QA pair

To test a single question / answer pair:

```bash
python app.py --config <OPTINAL_CUSTOM_MODEL_CONFIG> test --model <MODEL_NAME> --question "Is the sky blue?" --context "During a clear day, the sky appears blue
```

The response will be similar to the one from the API:

```json
{
  "name": "Clear Reply",
  "confidence": 0.95,
  "scores": {
    "Clear Reply": 0.95,
    "Clear Non-Reply": 0.03,
    "Ambivalent": 0.02
  }
}
```

### Dataset Evaluation

If you want to evaluate a whole test dataset, you can provide a JSON file with multiple entries in the same format as
described in the [Data Format](#data-format) section.

```bash
python app.py --config <OPTINAL_CUSTOM_MODEL_CONFIG> test --model <MODEL_NAME> --file <PATH_TO_TEST_FILE.json>
```

This will return evaluation metrics such as accuracy and F1-score for the provided dataset.

---

## Fine-Tuning with Together

You can fine-tune models hosted on Together.ai using the dashboard on the [Together Platform](https://together.ai/).
This enables way faster training and deployment of LLMs with a lot of parameters.

To prepare your dataset for Together fine-tuning, use the following format in a JSONL file. For more information
regarding
the dataset format, see the [README](../clarity-dataset/README.md#together-export-format) of
the `clarity-dataset` module.

```
{ "prompt": "<prompt-text-with-context-and-question>", "completion": "Label: <clarity_label>" }
{ "prompt": "<prompt-text-with-context-and-question>", "completion": "Label: <clarity_label>" }
{ "prompt": "<prçompt-text-with-context-and-question>", "completion": "Label: <clarity_label>" }
```

### Steps

1. Prepare your dataset in the Together format.
2. Log in to your Together account and navigate to the fine-tuning section.
3. Upload your dataset and configure the fine-tuning parameters.
4. Start the fine-tuning process and monitor its progress via the Together dashboard.

### Using Fine-Tuned Models

Before you start the testing you need to create a dedicated endpoint for your fine-tuned model on the Together platform.
To do so, follow the instructions in
the [Together Documentation](https://docs.together.ai/docs/fine-tuning-quickstart/).

Once your model is fine-tuned and deployed, you can use it in the `clarity-models` module by specifying the model name
in the
`together` model configuration.

### Configuration Example

For the initial experiments the following LoRA configuration was used:

- **Rank**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 0.0001
- **Max Gradient Norm**: 1
- **Weight Decay**: 0.005
- **Warmup Ratio**: 0.05
- **Scheduler Cycles**: 0.5
- **Checkpoints**: 1
- **Evaluations**: 1

