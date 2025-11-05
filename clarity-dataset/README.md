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

---

## Overview

Project structure (relevant files):

```yaml
├── data/                         # Output data directory (configurable via DATA_DIR)
├── Dockerfile                    # Docker image definition
├── app.py                        # Entrypoint: download, clean, split and Together-export
├── clean.py                      # Data cleaning and transformation helpers
├── summary.py                    # BERT-based summary generation
├── docker-compose.yaml           # Docker Compose configuration (mounts prompt file)
├── logging.yaml                  # Logging configuration
├── utils/                        # helper modules
├── requirements.txt              # Python dependencies
└── README.md               # This file
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

- **Dataset Download**  
  Downloads the dataset from the provided source.
  
- **Dataset Split**  
  Splits the training data into training and validation sets (80/20 split with random seed 42).
  
- **Data Transformation**  
  Simplifies and preprocesses the dataset for model consumption with configurable options:
  - Filler word removal (um, uh, you know, etc.)
  - President name removal (direct address, titles, full names)
  - BERT-based summary generation (dense vector embeddings)

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
# Generate BERT summaries (without cleaning)
docker compose run --rm dataset python app.py --use-bert
```

```bash
# Full processing: clean everything + BERT summaries
docker compose run --rm dataset python app.py --clean-all --use-bert
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
     - `summary_bert`: BERT embedding vector (if --use-bert enabled)

---

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

| Flag | Description |
|------|-------------|
| `--clean-fillers` | Remove filler words and phrases (um, uh, you know, etc.) |
| `--clean-names` | Remove president names and titles (Mr. President, Biden, etc.) |
| `--clean-all` | Apply both filler and name cleaning |
| `--use-bert` | Generate BERT-based summary embeddings for each QA pair using default BERT model|


**Supported BERT Model:**
- `bert-base-uncased` (default) - 768 dimensions

---

## Examples

### Generate multiple variants

```bash
# Variant 1: Original data with BERT
docker compose run --rm dataset python app.py --use-bert

# Variant 2: Clean fillers only
docker compose run --rm dataset python app.py --clean-fillers

# Variant 3: Full cleaning with BERT
docker compose run --rm dataset python app.py --clean-all --use-bert
```

### Custom workflow

```bash
# 1. Build the image
docker compose build

# 2. Run with specific configuration
docker compose run --rm dataset python app.py --clean-all --use-bert

# 3. Check the output
ls -lh ../data/cleaned/
```

---

## Notes

- The BERT summary generation can take around 2 hours, depending on hardware