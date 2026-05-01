package de.tum.claritypipeline;

import de.tum.claritypipeline.service.CategoryAccuracyEvaluator;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

/**
 * Integration tests for {@link CategoryAccuracyEvaluator}.
 *
 * <p>These tests connect to the live Neo4j instance (credentials loaded from the default
 * configuration file) and write Excel output to {@code src/test/resources/evaluation/}. They do not assert specific
 * numeric values; their purpose is to verify that the full query-to-Excel pipeline executes without errors and produces
 * a readable output file that can be inspected manually.
 *
 * <p>Prerequisite: the Neo4j database must be running and populated with the classification
 * runs referenced below.
 */
public class CategoryAccuracyEvaluatorTest {
    
    /**
     * The evaluator under test, connected to the default Neo4j instance.
     */
    private final CategoryAccuracyEvaluator evaluator = CategoryAccuracyEvaluator.create();
    
    /**
     * Default constructor.
     *
     * @throws IOException if the Neo4j credentials file cannot be read
     */
    public CategoryAccuracyEvaluatorTest() throws IOException { }
    
    /**
     * Exports per-category accuracy for the core evasion-based classification runs to Excel.
     *
     * <p>Covers the baseline GPT-5.2 evasion-based run as well as all 13 prompt repair
     * iterations (It.&nbsp;0–12) so that accuracy trends across iterations can be compared side-by-side in the
     * resulting workbook.
     *
     * <p>Output is written to {@code src/test/resources/evaluation/category_accuracy_ipr.xlsx}.
     *
     * @throws IOException if the Excel file cannot be written
     */
    @Test
    public void testExportPromptRepairIterations() throws IOException {
        List<CategoryAccuracyEvaluator.CategoryEvaluationOptions.ClassificationRun> runs = List.of(
                run("GPT 5.2", "reasoning-effort-auto:evasion-based:single:few-shot:v1"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-1:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-2:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-3:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-4:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-5:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-6:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-7:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-8:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-9:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-10:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-11:v3"),
                run("GPT 5.2", "single:few-shot:enhanced-prompt-iteration-12:v3")
        );
        
        evaluator.exportCategoryAccuracies(
                CategoryAccuracyEvaluator.CategoryEvaluationOptions.builder()
                        .classificationRuns(runs)
                        .excelPath("src/test/resources/evaluation/category_accuracy_ipr.xlsx")
                        .build()
        );
    }
    
    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    
    /**
     * Convenience factory that builds a {@link CategoryAccuracyEvaluator.CategoryEvaluationOptions.ClassificationRun}.
     *
     * @param name    the {@code ClassificationProperties.name} value in Neo4j
     * @param version the {@code ClassificationProperties.version} value in Neo4j
     * @return a fully constructed {@code ClassificationRun}
     */
    private CategoryAccuracyEvaluator.CategoryEvaluationOptions.ClassificationRun run(
            String name, String version) {
        return CategoryAccuracyEvaluator.CategoryEvaluationOptions.ClassificationRun.builder()
                .classificationName(name)
                .classificationVersion(version)
                .build();
    }
}
