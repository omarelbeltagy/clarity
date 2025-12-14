package de.tum.claritypipeline;

import de.tum.claritypipeline.model.config.DatasetType;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.service.ClassificationPipeline;
import de.tum.claritypipeline.service.DatasetGraphImporter;
import de.tum.claritypipeline.service.DatasetReader;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Tests the overall classification pipeline.
 *
 * <p>Includes dataset ingestion, graph import, and running classification experiments using
 * different model/agent configurations defined by properties files.</p>
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
     * Default constructor.
     *
     * @throws IOException if initialization fails
     */
    public PipelineTest() throws IOException {}

    /**
     * Import training, validation and test datasets into the graph importer.
     *
     * <p>This merges multiple dataset splits into a single in-memory collection and
     * imports them into the neo4j database.</p>
     */
    @Test
    public void testImportDatasets() {
        List<QA> data = new ArrayList<>();
        for (DatasetType datasetType : DatasetType.values()) {
            data.addAll(datasetReaderService.readDataset(datasetType));
        }
        datasetGraphImporter.importDataset(data);
    }

    @Test
    public void testClassifyFromDirectory() {
        final String baseDir = "src/test/resources/properties/stage1/";

        classifyFromDirectory(baseDir + "pag-few-shot");
        classifyFromDirectory(baseDir + "pag-few-shot-evasion-based");
        classifyFromDirectory(baseDir + "single-few-shot");
        classifyFromDirectory(baseDir + "single-few-shot-rag");
        classifyFromDirectory(baseDir + "single-few-shot-evasion-based-rag");
        classifyFromDirectory(baseDir + "single-few-shot-evasion-based-rag-reasoning-high");
        classifyFromDirectory(baseDir + "single-few-shot-reasoning-high");
        classifyFromDirectory(baseDir + "single-few-shot-evasion-based");
        classifyFromDirectory(baseDir + "single-few-shot-evasion-based-reasoning-high");
    }

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

    @Test
    public void testClassifyFromFile() throws IOException {
        String file = "src/test/resources/properties/stage2/evasion-based/gpt-5.2-reasoning-auto.yaml";
        ClassificationPipeline classificationPipeline = new ClassificationPipeline(file);
        classificationPipeline.classify();
    }
}
