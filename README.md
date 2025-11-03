# Unmasking Political Question Evasions

- [📊 SemEval 2026 Task](https://konstantinosftw.github.io/CLARITY-SemEval-2026/)
- [🤗 Dataset](https://huggingface.co/datasets/ailsntua/QEvasion)
- [📄 A dataset, taxonomy and baselines on response clarity classification](https://arxiv.org/abs/2409.13879)

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

* [Git](https://git-scm.com)
* [Docker](https://www.docker.com/get-started/)
* An IDE such as [IntelliJ IDEA](https://www.jetbrains.com/de-de/idea/)
  or [Eclipse](https://www.eclipse.org/downloads/) (optional but recommended)

---

## Setup

> To set up the project and run it locally, follow these steps:

TODO

---

## Modules

> The project is separated into several modules, each responsible for different functionalities:

TODO

---

## Pipeline

> The classification pipeline consists of several steps to process the data and classify the clarity of responses

TODO