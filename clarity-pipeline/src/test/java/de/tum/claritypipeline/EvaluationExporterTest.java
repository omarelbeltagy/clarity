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
     * Export evaluation results as an Excel file.
     *
     * <p>This test writes a sample or current evaluation to the given xlsx path. The produced
     * file should contain sheets/tables with evaluation metrics and per-example results.</p>
     */
    @Test
    public void testExportEvaluationToExcel() {
        evaluationExporter.exportAsExcel("src/test/resources/evaluation/12_02_2025.xlsx");
    }

    @Test
    public void testExportResult() throws IOException {
        evaluationExporter.exportResult(
                "src/test/resources/properties/single-few-shot-evasion-based-rag/gpt-5.1.yaml",
                "src/test/resources/prediction.zip"
        );
    }

    @Test
    public void testGenerateCustomEvaluation() throws IOException {
        evaluationExporter.generateCustomEvaluation(
                "src/test/resources/properties/single-few-shot-evasion-based-rag/gpt-5.1.yaml"
        );
    }
}
