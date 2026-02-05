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
- [Processing Options](#processing-options)
- [Together export format](#together-export-format)
- [Examples](#examples)
- [Spacy Cleaner](#spacy-cleaner)

---

## Overview

Project structure (relevant files):

```yaml
├── data/                         # Output data directory (configurable via DATA_DIR)
├── utils/                        # Helper modules
├── app.py                        # Entrypoint: download, clean, split and Together-export
├── clean.py                      # Data cleaning and transformation helpers
├── docker-compose.yaml           # Docker Compose configuration (mounts prompt file)
├── Dockerfile                    # Docker image definition
├── logging.yaml                  # Logging configuration
├── requirements.txt              # Python dependencies
├── spacy_cleaner.py              # SpaCy-based text cleaner
└── README.md                     # This file
```

---

## Features

- Dataset download from Hugging Face [(`ailsntua/QEvasion`](https://huggingface.co/datasets/ailsntua/QEvasion)).
- Train split is shuffled and split 80/20 into train / valid.
- Data cleaning: filler removal, bracket removal, and redaction of interviewee names.
- Exports:
    - Raw JSONs under [`full/`](data/full)
    - Cleaned JSONs under [`cleaned/`](data/cleaned)
    - Together-compatible JSONL under [`together/`](data/together) (used for Together API ingestion)

---

## Output & Directories

- **Dataset Download**  
  Downloads the dataset from the provided source.

- **Dataset Split**  
  Splits the training data into training and validation sets (80/20 split with random seed 42).

- **Data Transformation**  
  Simplifies and preprocesses the dataset for model consumption with configurable options:
    - Filler word removal (um, uh, you know, etc.)
    - President name removal (direct address, titles, full names)

- **Flexible Processing**  
  Control which preprocessing steps to apply via command-line flags.

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

### Prerequisites

* [Docker](https://www.docker.com/get-started/) (for containerized execution)
* [Python 3.8+](https://www.python.org/downloads/) (for native execution)

### Build and Run (Docker)

From inside the `clarity-dataset/` directory:

```bash
# Basic usage
docker compose up -d
```

```bash
# Clean fillers only
docker compose run --rm dataset python app.py --clean-fillers
```

```bash
# Clean names only
docker compose run --rm dataset python app.py --clean-names
```

```bash
# Clean everything (fillers + names)
docker compose run --rm dataset python app.py --clean-all
```

```bash
# Rebuild from scratch if needed
docker compose build --no-cache
```

### Output Structure

The script generates two sets of outputs:

1. **Full datasets** (`/data/full/`)
    - Original, unprocessed data
    - Files: `train.json`, `valid.json`, `test.json`

2. **Cleaned datasets** (`/data/cleaned/`)
    - Processed according to specified flags
    - Files: `train.json`, `valid.json`, `test.json`
    - Each item contains:
        - `question`: Original question
        - `context`: Original interview Q&A concatenated
        - `question_clean`: Cleaned question (if cleaning enabled)
        - `context_clean`: Cleaned context (if cleaning enabled)
        - `clarity_label`: Classification label

---

### Native

For native execution, from inside the `clarity-dataset/` directory:

```bash
python3 -m venv venv # Create virtual environment
source venv/bin/activate # Activate virtual environment
pip install -r requirements.txt # Install dependencies
python -m spacy download en_core_web_sm #download spacy manually
python app.py # Start the data processing

```

---

## Configuration

### Logging

- Logging
    - Configured via [logging.yaml](logging.yaml)
- Environment variables:
    - `DATA_DIR`
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

Entries written to `{DATA_DIR}/together/*.jsonl` follow the shape:

```
{ "prompt": "<prompt-text-with-context-and-question>", "completion": "Label: <clarity_label>" }
{ "prompt": "<prompt-text-with-context-and-question>", "completion": "Label: <clarity_label>" }
{ "prompt": "<prompt-text-with-context-and-question>", "completion": "Label: <clarity_label>" }
```

One JSON object per line (JSONL). Only records with non-empty interview_question, interview_answer, question and
clarity_label are exported.

### Command-Line Flags

| Flag              | Description                                                                      |
|-------------------|----------------------------------------------------------------------------------|
| `--clean-fillers` | Remove filler words and phrases (um, uh, you know, etc.)                         |
| `--clean-names`   | Remove president names and titles (Mr. President, Biden, etc.)                   |
| `--clean-all`     | Apply both filler and name cleaning                                              |


## Example

### Generate multiple variants

```bash
# Variant 1: Clean names only
docker compose run --rm dataset python app.py --clean-names

# Variant 2: Clean fillers only
docker compose run --rm dataset python app.py --clean-fillers

### Custom workflow

```bash
# 1. Build the image
docker compose build

# 2. Run with specific configuration
docker compose run --rm dataset python app.py --clean-all

# 3. Check the output
ls -lh ../data/cleaned/
```

## Spacy Cleaner

The `spacy_cleaner.py` module provides advanced text cleaning capabilities using the SpaCy library. It is used to remove
filler words and president names from the interview transcripts.

### Features

- Uses spaCy for intelligent tokenization and stopword handling
- Adds custom filler words and phrases
- Allows selective stopword preservation

---