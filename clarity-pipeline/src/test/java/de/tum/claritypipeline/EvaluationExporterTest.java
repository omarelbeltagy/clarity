package de.tum.claritypipeline;

import de.tum.claritypipeline.service.EvaluationExporter;
import org.junit.jupiter.api.Test;

import java.io.IOException;

/**
 * Tests evaluation export utilities.
 *
 * <p>Verifies that evaluation results can be exported to Excel format for manual inspection
 * or reporting purposes.</p>
 */
public class EvaluationExporterTest {
    /**
     * Service performing evaluation export operations.
     */
    private final EvaluationExporter evaluationExporter = EvaluationExporter.create();

    /**
     * Default constructor.
     *
     * @throws IOException if exporter initialization fails
     */
    public EvaluationExporterTest() throws IOException {}

    /**
     * Generates the Excel workbook that aggregates accuracy/precision/recall/F1 across classification runs.
     */
    @Test
    public void testExportEvaluationToExcel() {
        evaluationExporter.exportAsExcel("src/test/resources/evaluation/12_22_2025.xlsx");
    }

    /**
     * Writes a competition-ready ZIP (“prediction”) by reading the configured classification properties.
     */
    @Test
    public void testExportResult() throws IOException {
        evaluationExporter.exportResult(
                "src/test/resources/properties/stage1/single-few-shot-evasion-based-rag/gpt-5.1.yaml",
                "src/test/resources/prediction.zip"
        );
    }

    /**
     * Computes the evasion-level multi-label evaluation described in the README and persists it back to Neo4j.
     */
    @Test
    public void testGenerateCustomEvaluation() throws IOException {
        evaluationExporter.generateCustomEvaluation(
                "src/test/resources/properties/stage1/single-few-shot-evasion-based-rag-reasoning-high/gpt-5.1.yaml"
        );
        evaluationExporter.generateCustomEvaluation(
                "src/test/resources/properties/stage2/judgement/gpt-5.1-reasoning-high.yaml"
        );
    }
}
