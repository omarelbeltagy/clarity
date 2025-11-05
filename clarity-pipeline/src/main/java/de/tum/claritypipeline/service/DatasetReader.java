package de.tum.claritypipeline.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.tum.claritypipeline.model.QA;
import de.tum.claritypipeline.model.properties.DatasetType;
import lombok.NoArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;

/**
 * Service for reading the dataset from JSON files.
 */
@NoArgsConstructor
public class DatasetReader {
    private static final Logger log = LoggerFactory.getLogger(DatasetReader.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private String basePath = "../clarity-dataset/data/full";

    public DatasetReader(String basePath) {
        this.basePath = basePath;
    }

    public List<QA> readDataset(DatasetType datasetType) {
        return switch (datasetType) {
            case TEST -> readDatasetHelper("test.json", DatasetType.TEST);
            case VALID -> readDatasetHelper("valid.json", DatasetType.VALID);
            case TRAIN -> readDatasetHelper("train.json", DatasetType.TRAIN);
            default -> readDatasetHelper("unknown.json", DatasetType.GENERIC);
        };
    }

    public List<QA> readDataset(String fileName, DatasetType type) {
        return readDatasetHelper(fileName, type);
    }

    public List<QA> readDataset(String fileName) {
        return readDatasetHelper(fileName, DatasetType.GENERIC);
    }

    private List<QA> readDatasetHelper(String fileName, DatasetType type) {
        Path filePath = Paths.get(basePath, fileName);
        try {
            List<QA> records = MAPPER.readValue(filePath.toFile(), new TypeReference<>() {});
            records.forEach(record -> applyDatasetType(record, type));
            log.info("Loaded {} records from {}", records.size(), fileName);
            return records;
        } catch (IOException e) {
            log.error("Error reading dataset from {}: {}", fileName, e.getMessage(), e);
            return Collections.emptyList();
        }
    }

    private void applyDatasetType(QA record, DatasetType type) {
        record.setTest(type == DatasetType.TEST);
        record.setValid(type == DatasetType.VALID);
        record.setTrain(type == DatasetType.TRAIN);
    }
}