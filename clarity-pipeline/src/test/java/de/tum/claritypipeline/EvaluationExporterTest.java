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
        evaluationExporter.exportAsExcel("src/test/resources/evaluation/11_04_2025_INITIAL_EXPERIMENTS.xlsx");
    }
}
