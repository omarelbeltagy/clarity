# Clarity Models

> This module provides a dockerized service for running different classification models through a **FastAPI**
> interface.  
> It loads multiple models defined in a configuration file and expose them as REST endpoints.

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
├── app.py # Entrypoint defining FastAPI app and loading models
├── docker-compose.yaml # Docker Compose configuration
├── logging.yaml # Loguru logging configuration
├── models # Directory containing model implementations
│ └── roberta_base.py
│ └── ...
├── models.yaml # Configuration file defining which models to load
└── requirements.txt # Python dependencies
```

### Features

- **Multiple model deployment**  
  Models are defined in [models.yaml](models.yaml) and loaded automatically at startup.
- **Endpoints**  
  Each model is exposed as an HTTP endpoint for inference.

---

## Usage

### Prerequisites

* [Docker](https://www.docker.com/get-started/)

### Build and Run

From inside the `clarity-models/` directory:

```bash
# Build and start the container
docker compose up -d
```

```bash
# Rebuild from scratch if needed
docker compose build --no-cache
```

### Accessing the Service

The service will be available at: http://localhost:8078

#### Usage Example Endpoint

The default roberta-base model is available at: `POST /classify/roberta-base`

``` bash
curl -X POST "http://localhost:8078/classify/roberta-base" \
-H "Content-Type: application/json" \
-d '{ "question": "Do you like taxes?", "answer": "We must think about our economy." }'
```

The expected response is similar to:

```json
{
  "clarity_label": "Ambivalent",
  "confidence": 0.69,
  "scores": {
    "Clear Reply": 0.05,
    "Clear Non-Reply": 0.26,
    "Ambivalent": 0.69
  }
}
```

---

## Configuration

* [models.yaml](models.yaml) Defines which models are loaded and their routes
* [logging.yaml](logging.yaml) Configures logging output