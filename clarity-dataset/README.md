# Clarity Dataset

- [🤗 Dataset](https://huggingface.co/datasets/ailsntua/QEvasion)

This module handles the download, cleaning and transformation of the dataset used for the Clarity classification task
and produces exports compatible with the Together API.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Output & Directories](#output--directories)
- [Usage](#usage)
- [Configuration](#configuration)

---

## Overview

Project structure (relevant files):

```yaml
├── data/                         # Output data directory (configurable via DARA_DIR)
├── Dockerfile                    # Docker image definition
├── app.py                        # Entrypoint: download, clean, split and Together-export
├── cleaning.py                   # Data cleaning and transformation helpers
├── docker-compose.yaml           # Docker Compose configuration (mounts prompt file)
├── logging.yaml                  # Logging configuration
├── utils/                        # helper modules
└── requirements.txt              # Python dependencies
```

---

## Features

- Dataset download from Hugging Face (`ailsntua/QEvasion`).
- Train split is shuffled and split 80/20 into train / valid.
- Data cleaning: filler removal, bracket removal, and redaction of interviewee names.
- Exports:
    - Raw JSONs under `full/`
    - Cleaned JSONs under `cleaned/`
    - Together-compatible JSONL under `together/` (used for Together API ingestion)

---

## Output & Directories

By default the tool writes to a data directory (see [Configuration](#configuration)). The following structure will be
created:

- [data/full/](data/full)
    - [train.json](data/full/train.json), [valid.json](data/full/valid.json), [test.json](data/full/test.json)
- [data/cleaned/](data/cleaned)
    - [train.json](data/cleaned/train.json), [valid.json](data/cleaned/valid.json), [test.json](data/cleaned/test.json)
- [data/together/](data/together)
    - [train.jsonl](data/together/train.jsonl), [valid.jsonl](data/together/valid.jsonl)

Together JSONL entries are generated from a [prompt template](../assets/prompts/lora.yaml) loaded from a YAML file
(`LORA_PROMPT_FILE`).

---

## Usage

Prerequisites:

- Docker

Build and run (from clarity-dataset/):

```bash
# Build and start container (daemon)
docker compose up -d

# Rebuild without cache if needed
docker compose build --no-cache
```

### Prerequisites

* [Docker](https://www.docker.com/get-started/) (for containerized execution)
* [Python 3.8+](https://www.python.org/downloads/) (for native execution)

### Build and Run (Docker)

From inside the `clarity-dataset/` directory:

```bash
# Build and start the container
docker compose up -d
```

```bash
# Rebuild from scratch if needed
docker compose build --no-cache
```

### Native

For native execution, from inside the `clarity-dataset/` directory:

```bash
python3 -m venv venv # Create virtual environment
source venv/bin/activate # Activate virtual environment
pip install -r requirements.txt # Install dependencies
python app.py # Start the data processing
```

---

## Configuration

- Logging
    - Configured via [logging.yaml](logging.yaml)
- Environment variables:
    - `DARA_DIR`
        - Base directory where outputs are written.

    - `LORA_PROMPT_FILE`
        - Path to a YAML file containing the prompt template under the key `prompt`.
        - Expected YAML format:
          ```yaml
          prompt: |
            <PROMPT_CONTENT>
            <CONTEXT_PLACEHOLDER>: {context}
            <QUESTION_PLACEHOLDER>: {question}
          ```

---

## Together export format

Entries written to `{DARA_DIR}/together/*.jsonl` follow the shape:

```
{
    "prompt": "<prompt-text-with-context-and-question>",
    "completion": "Label: <clarity_label>"
}
```

One JSON object per line (JSONL). Only records with non-empty interview_question, interview_answer, question and
clarity_label are exported.
