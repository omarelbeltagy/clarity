package de.tum.claritypipeline.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.opencsv.bean.CsvToBeanBuilder;
import de.tum.claritypipeline.model.config.DatasetType;
import de.tum.claritypipeline.model.core.QA;
import lombok.NoArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;

/**
 * Service for reading QA datasets from JSON files.
 * <p>
 * This service provides flexible dataset loading capabilities for the Clarity pipeline,
 * supporting both standard split files (train/valid/test) and custom JSON files.
 * It automatically applies dataset type metadata to loaded records.
 *
 * <h2>Dataset Structure</h2>
 * The service expects JSON files containing arrays of QA objects with structure matching
 * the {@link QA} model. Standard dataset splits are:
 * <ul>
 *   <li><b>train.json</b>: Training data for model development</li>
 *   <li><b>valid.json</b>: Validation data for hyperparameter tuning</li>
 *   <li><b>test.json</b>: Test data for final evaluation</li>
 * </ul>
 *
 * <h2>Dataset Type Tagging</h2>
 * Each loaded QA record is automatically tagged with boolean flags indicating
 * its dataset split membership:
 * <ul>
 *   <li><b>test</b>: true for test split records</li>
 *   <li><b>valid</b>: true for validation split records</li>
 *   <li><b>train</b>: true for training split records</li>
 * </ul>
 * This metadata enables split-based filtering in Cypher queries.
 *
 * <h2>Base Path Configuration</h2>
 * The service uses a configurable base path for locating JSON files:
 * <ul>
 *   <li>Default: {@code ../clarity-dataset/data/full}</li>
 *   <li>Customizable via constructor</li>
 *   <li>Relative or absolute paths supported</li>
 * </ul>
 *
 * <h2>Error Handling</h2>
 * <ul>
 *   <li>Returns empty list on read failures (doesn't throw exceptions)</li>
 *   <li>Logs detailed error information including file path and exception details</li>
 *   <li>Suitable for graceful degradation in pipeline workflows</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>
 * // Standard splits
 * DatasetReader reader = new DatasetReader();
 * List&lt;QA&gt; trainData = reader.readDataset(DatasetType.TRAIN);
 * List&lt;QA&gt; testData = reader.readDataset(DatasetType.TEST);
 *
 * // Custom file
 * List&lt;QA&gt; customData = reader.readDataset("custom-split.json", DatasetType.GENERIC);
 *
 * // Custom base path
 * DatasetReader customReader = new DatasetReader("/path/to/datasets");
 * List&lt;QA&gt; data = customReader.readDataset(DatasetType.VALID);
 * </pre>
 *
 * @see QA
 * @see DatasetType
 * @see DatasetGraphImporter
 */
@NoArgsConstructor
public class DatasetReader {
    private static final Logger log = LoggerFactory.getLogger(DatasetReader.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private String basePath = "../clarity-dataset/data/full";

    /**
     * Constructs a DatasetReader with a custom base path for dataset files.
     * <p>
     * The base path is prepended to all file names when loading datasets.
     *
     * @param basePath the directory path containing JSON dataset files
     */
    public DatasetReader(String basePath) {
        this.basePath = basePath;
    }

    /**
     * Reads a standard dataset split based on the dataset type.
     * <p>
     * Maps dataset types to their corresponding JSON files:
     * <ul>
     *   <li>{@link DatasetType#TEST} → test.json</li>
     *   <li>{@link DatasetType#VALID} → valid.json</li>
     *   <li>{@link DatasetType#TRAIN} → train.json</li>
     *   <li>{@link DatasetType#GENERIC} → unknown.json</li>
     * </ul>
     *
     * @param datasetType the type of dataset split to load
     * @return list of QA records with appropriate dataset type flags set,
     * or empty list if file cannot be read
     */
    public List<QA> readDataset(DatasetType datasetType) {
        return switch (datasetType) {
            case TEST -> readDatasetHelper("test.json", DatasetType.TEST);
            case VALID -> readDatasetHelper("valid.json", DatasetType.VALID);
            case TRAIN -> readDatasetHelper("train.json", DatasetType.TRAIN);
            case EVALUATION -> readDatasetHelper("../evaluation/evaluation.csv", DatasetType.EVALUATION);
            default -> readDatasetHelper("unknown.json", DatasetType.GENERIC);
        };
    }

    /**
     * Reads a dataset from a custom JSON file with explicit type tagging.
     * <p>
     * Useful for loading non-standard splits or custom dataset variations
     * while maintaining proper dataset type metadata.
     *
     * @param fileName the name of the JSON file (relative to base path)
     * @param type     the dataset type to assign to all loaded records
     * @return list of QA records with specified dataset type flags set,
     * or empty list if file cannot be read
     */
    public List<QA> readDataset(String fileName, DatasetType type) {
        return readDatasetHelper(fileName, type);
    }

    /**
     * Reads a dataset from a custom JSON file with generic type tagging.
     * <p>
     * Convenience method for loading custom files when dataset type
     * classification is not important. All records will have no split flags set.
     *
     * @param fileName the name of the JSON file (relative to base path)
     * @return list of QA records with generic (no split) type flags,
     * or empty list if file cannot be read
     */
    public List<QA> readDataset(String fileName) {
        return readDatasetHelper(fileName, DatasetType.GENERIC);
    }

    /**
     * Internal helper method for loading and parsing JSON dataset files.
     * <p>
     * Handles:
     * <ul>
     *   <li>File path resolution (base path + file name)</li>
     *   <li>JSON deserialization via Jackson ObjectMapper</li>
     *   <li>Dataset type flag application to all records</li>
     *   <li>Logging of successful loads and errors</li>
     * </ul>
     *
     * @param fileName the JSON file name
     * @param type     the dataset type to assign
     * @return list of loaded QA records or empty list on failure
     */
    private List<QA> readDatasetHelper(String fileName, DatasetType type) {
        Path filePath = Paths.get(basePath, fileName);

        try {
            List<QA> records;

            if (fileName.toLowerCase().endsWith(".csv")) {
                records = readFromCsv(filePath);
            } else {
                records = readFromJson(filePath);
            }

            records.forEach(record -> applyDatasetType(record, type));
            log.info("Loaded {} records from {}", records.size(), fileName);
            return records;

        } catch (Exception e) {
            log.error("Error reading dataset from {}: {}", fileName, e.getMessage(), e);
            return Collections.emptyList();
        }
    }

    private List<QA> readFromJson(Path filePath) throws IOException {
        return MAPPER.readValue(filePath.toFile(), new TypeReference<List<QA>>() {});
    }

    private List<QA> readFromCsv(Path filePath) throws IOException {
        try (Reader reader = new FileReader(filePath.toFile())) {
            return new CsvToBeanBuilder<QA>(reader)
                    .withType(QA.class)
                    .withIgnoreLeadingWhiteSpace(true)
                    .build()
                    .parse();
        }
    }


    /**
     * Applies dataset type flags to a QA record.
     * <p>
     * Sets boolean flags on the QA object indicating which dataset split
     * it belongs to. Only the flag corresponding to the specified type
     * is set to true; all others are false.
     *
     * @param record the QA record to tag
     * @param type   the dataset type determining which flags to set
     */
    private void applyDatasetType(QA record, DatasetType type) {
        record.setTest(type == DatasetType.TEST);
        record.setValid(type == DatasetType.VALID);
        record.setTrain(type == DatasetType.TRAIN);
        record.setEvaluation(type == DatasetType.EVALUATION);
    }
}