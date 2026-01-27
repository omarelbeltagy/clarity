# Unmasking Political Question Evasions

- [📝 Pipeline Documentation](clarity-pipeline/README.md)
- [📊 SemEval 2026 Task](https://konstantinosftw.github.io/CLARITY-SemEval-2026/)
- [🤗 Dataset](https://huggingface.co/datasets/ailsntua/QEvasion)
- [🤗 Evaluation Dataset](https://github.com/konstantinosftw/CLARITY-SemEval-2026/blob/main/dataset/clarity_task_evaluation_dataset.csv)
- [📄 A dataset, taxonomy and baselines on response clarity classification](https://arxiv.org/abs/2409.13879)
- [:octocat: GitHub Repository](https://github.com/omarelbeltagy/clarity)

> A repository for the SemEval 2026 Task on Unmasking Political Question Evasions.

---

## Table of Contents

- [Overview](#overview)
    - [Topic](#topic)
    - [Dataset](#dataset)
    - [Tasks](#tasks)
    - [Evaluation](#evaluation)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
    - [Project Building](#project-building)
    - [Configuration](#api-keys)
    - [Database Setup](#database-setup)
- [Modules](#modules)
- [Pipeline](#pipeline)

---

## Overview

### Topic

This repository focuses on the task of classifying the clarity of political question responses.
Given a question and an answer, the goal is to determine whether the answer is clear, ambiguous, or a non-reply,
and if ambiguous, to identify the specific evasion technique used.

### Dataset

The [dataset](https://huggingface.co/datasets/ailsntua/QEvasion) consists of question / answer pairs from presidential
interviews and is separated into training and test sets.

### Tasks

#### Task 1 – Clarity‐level Classification

A coarse classification into clear reply / ambiguous / clear non‐reply.

- **Given**: a question / answer pair.
- **Output**: one of {Clear Reply, Ambiguous, Clear Non­Reply}.

#### Task 2 – Evasion‐level Classification

A fine‐grained classification into nine distinct evasion techniques (for cases of ambiguity).

- **Given**: the same question / answer pair.
- **Output**: one of nine evasion types (only applicable when answer is ambiguous/evasive) or a “no evasion” label
  depending on setup.

### Evaluation

Both tasks use macro F1‐score evaluated on a test set.

---

## Prerequisites

> Before setting up the project, ensure you have the following installed:

* [Java 21](https://www.oracle.com/de/java/technologies/downloads/)
* [Maven 3.9.9](https://maven.apache.org/download.cgi)
* [Git](https://git-scm.com)
* [Docker](https://www.docker.com/get-started/)
* [Python 3.9+](https://www.python.org/downloads/)
* An IDE such as [IntelliJ IDEA](https://www.jetbrains.com/de-de/idea/)
  or [Eclipse](https://www.eclipse.org/downloads/) (optional but recommended)

---

## Setup

> To set up the project and run it locally, follow these steps:

### Project Building

1. Clone the repository:
    ```bash
    git clone https://github.com/omarelbeltagy/clarity.git
    ```

2. Build the java project with Maven:
    ```bash
    cd clarity
    mvn clean install -U -DskipTests
    ```

### API Keys

For using the APIs of common LLMs that are needed for the classification you need to provide API keys. For that create a
`.env` file in the project root with the following content:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
TOGETHER_API_KEY=your_together_api_key
```

### Database Setup

The project uses a [Neo4j](https://neo4j.com) graph database to store and manage data. It already provides
a [Neo4j docker-compose setup](clarity-neo4j/src/main/resources/docker-compose.yaml) for local development.

To start the Neo4j database locally in a Docker container:

1. Open a Terminal Window
2. Navigate to the [`docker-compose.yaml`](clarity-neo4j/src/main/resources/docker-compose.yaml) file for neo4j
    ```bash
    cd <PROJECT_ROOT>
    cd clarity-neo4j/src/main/resources
    ```
3. Start the docker container:
    ```bash
    docker compose up -d
    ```

The `docker-compose` file is just a helper for local development. You can use any neo4j instance and configure the
connection details using a credentials yaml file and load the properties from
there [@see](clarity-neo4j/src/main/resources/neo4j-credentials.yaml).

---

## Modules

> The project is separated into several modules, each responsible for different functionalities:

- [**assets**](assets/README.md): Contains various assets used across the project, including
  prompt templates and the taxonomy for classification.
- [**clarity-dataset**](clarity-dataset/README.md): Download cleaning and transforming the dataset
  for clarity classification.
- [**clarity-models**](clarity-models/README.md): Framework for training and serving classification models. Supports
  encoder models and LLMs with LoRA Fine-Tuning. Provides instructions to train larger models with Together.ai.
- [**clarity-neo4j**](clarity-neo4j): Java utilities to handle interactions with the Neo4j database.
- [**clarity-pipeline**](clarity-pipeline/README.md): Contains the services and logic used in the classification
  pipeline. This
  includes data ingestion, processing of data and the actual classification using different approaches as well as
  automatic prompt engineering.
- [**clarity-utils**](clarity-utils): Contains general java utility classes and functions used across the project.

---

## Pipeline

> The classification pipeline consists of several steps to process the data and classify the clarity of responses

For detailed information on the pipeline and how to run it, refer to the
[README](clarity-pipeline/README.md) in the `clarity-pipeline` module.