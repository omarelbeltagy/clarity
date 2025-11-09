# Clarity Pipeline

> End-to-end orchestration layer for **LLM classification experiments**.  
> Integrates Neo4j, multi-provider LLMs, and evaluation tooling under one unified Java module.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Usage](#usage)
- [APIs](#apis)

---

## Overview

The **Clarity Pipeline** forms the backbone of the Clarity classification framework.

It provides an extensible runtime that connects data, ontology, model inference, and evaluation:

- Synchronizes **QA datasets** and **taxonomies** with a Neo4j ontology
- Interfaces with multiple **LLM providers** (OpenAI, Anthropic, Together, local inference)
- Orchestrates **prompt-based classification** and **judgement workflows**
- Stores **structured predictions**, explanations, and metadata as graph entities
- Aggregates and exports **evaluation metrics** (accuracy, precision, recall, macro/micro F1)

---

## Key Features

- **Unified orchestration layer**  
  Complete flow from data ingestion to evaluation.

- **Multi-provider LLM support**  
  Modular clients for OpenAI, Anthropic, Together, or local endpoints.

- **Graph-centric persistence**  
  Ontology management for clusters, taxonomies, categories, and runs in Neo4j.

- **Config-driven execution**  
  All behaviour controlled via version-tracked YAML descriptors.

- **Evaluation & export suite**  
  Aggregated metrics exportable as `.xlsx` or `.csv`.

- **Parallel and fault-tolerant**  
  Thread-pooled execution with automatic retries and logging.

- **Research-grade provenance**  
  Every run is reproducible, versioned, and ontology-linked.

---

## Requirements

- **R1 – End-to-End Classification Flow**
    - The system shall perform full inference runs from dataset loading to evaluation output in a single orchestrated
      process.
- **R2 - Persistence in Database**
    - The system shall store datasets, taxonomies, model configurations, predictions, and evaluations in database for
      future retrieval and analysis.
- **R3 - Multi-Provider LLM Support**
    - The system shall support multiple LLM providers (e.g., OpenAI, Anthropic, Together, locally hosted) through a
      unified client interface.
- **R4 - Config Driven Execution**
    - The system shall allow all aspects of the classification pipeline to be configured via external YAML files,
      enabling reproducible experiments and dynamic adjustments without code changes.

---

## Architecture

![Pipeline Diagram](assets/pipeline.png)

| Layer                      | Responsibility                                                                                       |
|----------------------------|------------------------------------------------------------------------------------------------------|
| **Service Layer**          | Orchestrates the end-to-end flow (`ClassificationPipeline`, `EvaluationExporter`, `DatasetImporter`) |
| **Client Layer**           | Connects to LLM providers via a unified `Client` contract                                            |
| **Strategy Layer**         | Defines classification logic (`SingleStrategy`, `JudgementStrategy`)                                 |
| **Model Configs**          | Encapsulate provider-specific options, prompts, and sampling params                                  |
| **Ontology Layer (Neo4j)** | Stores QA data, taxonomies, runs, and evaluation metrics                                             |
| **Utility Layer**          | Shared helpers for prompts, embeddings, and data I/O                                                 |

---

## Configuration

All runs are defined declaratively through YAML:

- **`ClassificationProperties`**  
  Run metadata: name, version, taxonomy, cluster, query, threading, strategy.
- **`ModelProperties`**  
  Provider, prompt templates, sampling params, response format.
- **`Taxonomy` YAMLs**  
  Category structures used to build the ontology.
- **`EvaluationExportProperties`**  
  Controls metric aggregation and export formatting.

YAML configs enable **reproducible experiments** and **consistent benchmarking** across models and datasets.

---

## Usage

### 1. Prepare Your Environment

- Start a Neo4j instance and ensure credentials in `clarity-neo4j` are valid
- Add API keys for your selected LLM providers
- Import datasets from `clarity-dataset` (`train`, `valid`, `test`)

### 2. Configure a Run

Define a YAML configuration (for example: `configs/few-shot/gpt-4.1.yaml`)  
that describes your cluster, taxonomy, and model setup.

```yaml
name: "gpt-4.1"
cluster: "OpenAI"
version: "few-shot:v1"
taxonomy: "../assets/taxonomy/clarity-categories.yaml"
neo4j-credentials: "../clarity-neo4j/src/main/resources/neo4j-credentials.yaml"
worker-threads: 16
query: |
  MATCH (n:QA)
  WHERE n.test = true
  RETURN n
strategy:
  type: "single"
  model:
    provider: "openai"
    name: "gpt-4.1"
    prompt: "../assets/prompts/few-shot.yaml"
    response-format: "json_object"
    max-tokens: 2048
    temperature: 0.2
```

### 3. Execute the Pipeline

Run via Maven or directly from Java:

```bash
mvn -pl clarity-pipeline test -Dtest=PipelineTest#testClassify
```

### 4. Export Results

Generate evaluation summaries as Excel workbooks:

```bash
mvn -pl clarity-pipeline test -Dtest=EvaluationExporterTest#testExportAsExcel
```

---

## APIs

| Category             | Main Classes / Interfaces                                          | Description                              |
|----------------------|--------------------------------------------------------------------|------------------------------------------|
| **Pipeline Control** | `ClassificationPipeline`                                           | Entry point for full classification runs |
| **Strategies**       | `SingleStrategy`, `JudgementStrategy`                              | Configurable inference behaviours        |
| **Clients**          | `OpenAIClient`, `AnthropicClient`, `TogetherClient`, `LocalClient` | Provider abstraction layer               |
| **Evaluation**       | `ModelEvaluator`, `EvaluationExporter`                             | Metric computation and reporting         |
| **Utilities**        | `PromptUtils`,                                                     | Prompt templates                         |        
