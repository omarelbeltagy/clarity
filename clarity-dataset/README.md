# Clarity Dataset

- [🤗 Dataset](https://huggingface.co/datasets/ailsntua/QEvasion)

> This module handles the download and transformation of the dataset provided for the clarity classification task

---

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Configuration](#configuration)

---

## Overview

### Project structure:

``` yaml
├── Dockerfile # Dockerfile to build the service container
├── app.py # Entrypoint containing the c
├── docker-compose.yaml # Docker Compose configuration
├── logging.yaml # Loguru logging configuration
└── requirements.txt # Python dependencies
```

### Features

- **Dataset Download**  
  Downloads the dataset from the provided source.
- **Dataset Split**  
  Splits the training data into training and validation sets.
- **Data Transformation**
  Simplifies and preprocesses the dataset for model consumption.

---

## Usage

### Prerequisites

* [Docker](https://www.docker.com/get-started/)

### Build and Run

From inside the `clarity-dataset/` directory:

```bash
# Build and start the container
docker compose up -d
```

```bash
# Rebuild from scratch if needed
docker compose build --no-cache
```

---

## Configuration

* [logging.yaml](logging.yaml) Configures logging output