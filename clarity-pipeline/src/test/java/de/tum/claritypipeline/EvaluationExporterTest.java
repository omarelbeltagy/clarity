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
        evaluationExporter.exportAsExcel("src/test/resources/evaluation/01_30_2026.xlsx");
    }

    /**
     * Writes a competition-ready ZIP (“prediction”) by reading the configured classification properties.
     */
    @Test
    public void testExportResult() throws IOException {
        evaluationExporter.exportResult(
                "src/test/resources/properties/evaluation/v2.yaml",
                "src/test/resources/prediction-v2-evasion..zip"
        );
        evaluationExporter.exportResult(
                "src/test/resources/properties/evaluation/v3.yaml",
                "src/test/resources/prediction.zip"
        );
    }

    /**
     * Computes the evasion-level multi-label evaluation described in the README and persists it back to Neo4j.
     */
    @Test
    public void testGenerateCustomEvaluation() throws IOException {
        evaluationExporter.generateCustomEvaluation(
                "src/test/resources/properties/stage2/evasion-based/gpt-5.2-reasoning-auto.yaml"
        );
    }
}
