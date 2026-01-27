package de.tum.claritypipeline;

import de.tum.claritypipeline.model.config.DatasetType;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.service.ClassificationPipeline;
import de.tum.claritypipeline.service.CleanedDataImporter;
import de.tum.claritypipeline.service.DatasetGraphImporter;
import de.tum.claritypipeline.service.DatasetReader;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * End-to-end tests mirroring the README “Usage” section: dataset ingestion, graph persistence, and strategy execution.
 */
public class PipelineTest {
    /**
     * Service for reading dataset JSON files from test resources.
     */
    private final DatasetReader datasetReaderService = new DatasetReader();

    /**
     * Service for importing QA records into an internal graph representation.
     */
    private final DatasetGraphImporter datasetGraphImporter = new DatasetGraphImporter();

    /**
     * Service for importing cleaned QA records into the graph database.
     */
    private final CleanedDataImporter cleanedDataImporter = new CleanedDataImporter();


    /**
     * Default constructor.
     *
     * @throws IOException if initialization fails
     */
    public PipelineTest() throws IOException {}

    /**
     * Reads every dataset split (train/valid/test) and imports them so the ontology layer mirrors the raw dataset.
     */
    @Test
    public void testImportDatasets() {
        List<QA> data = new ArrayList<>();
        for (DatasetType datasetType : DatasetType.values()) {
            data.addAll(datasetReaderService.readDataset(datasetType));
        }
        System.out.println("Imported " + data.size() + " QA pairs from all dataset splits.");
        //datasetGraphImporter.importDataset(data);
    }

    @Test
    public void testImportEvaluationDataset() {
        List<QA> data = datasetReaderService.readDataset(DatasetType.EVALUATION);
        System.out.println("Imported " + data.size() + " QA pairs from evaluation dataset.");
        datasetGraphImporter.importDataset(data);
    }

    /**
     * Imports cleaned variants to update existing QA nodes with normalized text, validating the cleaned-data workflow.
     */
    @Test
    public void testImportCleanedDatasets() {
        List<QA> data = new ArrayList<>();

        final DatasetReader datasetReaderCleaned = new DatasetReader("../clarity-dataset/data/cleaned/");

        data.addAll(datasetReaderCleaned.readDataset("test.json", DatasetType.TEST));
        data.addAll(datasetReaderCleaned.readDataset("train.json"));

        cleanedDataImporter.importCleanedData(data);
    }

    /**
     * Iterates over a directory of YAML configs and runs the classification pipeline exactly as described in
     * “Execute the Pipeline”.
     */
    @Test
    public void testClassifyFromDirectory() {
        final String baseDir = "src/test/resources/properties/stage2/enhanced-prompt/gpt-5.2/";

        classifyFromDirectory(baseDir);
        /*
        classifyFromDirectory(baseDir + "pag-few-shot");
        classifyFromDirectory(baseDir + "pag-few-shot-evasion-based");
        classifyFromDirectory(baseDir + "single-few-shot");
        classifyFromDirectory(baseDir + "single-few-shot-rag");
        classifyFromDirectory(baseDir + "single-few-shot-evasion-based-rag");
        classifyFromDirectory(baseDir + "single-few-shot-evasion-based-rag-reasoning-high");
        classifyFromDirectory(baseDir + "single-few-shot-reasoning-high");
        classifyFromDirectory(baseDir + "single-few-shot-evasion-based");
        classifyFromDirectory(baseDir + "single-few-shot-evasion-based-reasoning-high");
         */
    }

    /**
     * Helper that instantiates {@link ClassificationPipeline} for each YAML file and triggers the full inference +
     * evaluation loop.
     */
    private void classifyFromDirectory(String dirPath) {
        final int attempts = 1;
        File dir = new File(dirPath);
        File[] files = dir.listFiles();

        if (files != null) {
            for (int i = 0; i < attempts; i++) {
                Arrays.stream(files).forEach(file -> {
                    try {
                        ClassificationPipeline cp = new ClassificationPipeline(file.getAbsolutePath());
                        cp.classify();
                    } catch (IOException ex) {
                        throw new RuntimeException(ex);
                    }
                });
            }
        }
    }

    /**
     * Launches a single properties file to validate an individual pipeline configuration (useful for local debugging).
     */
    @Test
    public void testClassifyFromFile() throws IOException {
        String file = "src/test/resources/properties/stage1/single-few-shot-evasion-based-rag-reasoning-high/gpt-5.2"
                + ".yaml";
        ClassificationPipeline classificationPipeline = new ClassificationPipeline(file);
        classificationPipeline.classify();
    }
}
