# Pipeline Tests

> This README gives an overview of the tests for the pipeline components.

---

## Classes

- [DatabaseExporter](#DatabaseExporter)
- [EvaluationExporter](#EvaluationExporter)
- [Pipeline](#Pipeline)

---

## DatabaseExporter

- [♨️ DatabaseExporterTest.java](DatabaseExporterTest.java)

> Tests database export and import utilities.

- `testClearDatabase()`: Deletes all nodes and relationships from the Neo4j database to produce a clean state for other
  tests.
- `testExportDatabase()`: Exports the entire graph (nodes, relationships, properties) to a JSON file for backup or
  analysis.
- `testImportDatabase()`: Imports nodes and relationships from a JSON export back into Neo4j to restore a saved state.

## EvaluationExporter

- [♨️ EvaluationExporterTest.java](EvaluationExporterTest.java)

> Tests evaluation export utilities for producing reports.

- `testExportEvaluationToExcel()`: Exports evaluation results to an XLSX file. The file typically contains per-example
  annotations, metrics and summary sheets suitable for manual review or reporting.

## Pipeline

- [♨️ PipelineTest.java](PipelineTest.java)

> Tests the overall pipeline execution including data ingestion, processing, graph import and running classification
> experiments across multiple model configurations.

- `testImportDatasets()`: Reads TRAIN / VALID / TEST splits and imports them into the dataset graph importer for
  downstream processing.
- `testClassify()`: Runs a single classification pipeline configured via `maestro-reasoning.yaml` (example few-shot
  setup).

---
Notes:

- All test methods are integration-style and depend on local resources (properties, exported JSON, model access). Ensure
  credentials and resource files exist before running.
