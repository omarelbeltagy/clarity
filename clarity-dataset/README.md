# Clarity Dataset

- [🤗 Dataset](https://huggingface.co/datasets/ailsntua/QEvasion)

> This module handles the download and transformation of the dataset provided for the clarity classification task

---

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Configuration](#configuration)
- [Processing Options](#processing-options)

---

## Overview

### Project structure:

``` yaml
├── Dockerfile              # Dockerfile to build the service container
├── app.py                  # Entrypoint containing the c
├── clean.py                # Text cleaning utilities (fillers, names, etc.)
├── summary.py              # BERT-based summary generation
├── docker-compose.yaml     # Docker Compose configuration
├── logging.yaml            # Loguru logging configuration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Features

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

---

## Usage

### Prerequisites

* [Docker](https://www.docker.com/get-started/)

### Build and Run

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

## Configuration

### Logging

* [logging.yaml](logging.yaml) - Configures logging output using Loguru

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
