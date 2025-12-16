package de.tum.claritypipeline.service;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.claritypipeline.model.core.QA;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Service for importing QA datasets into a Neo4j graph database.
 * <p>
 * This service handles the initial import of question-answer pairs from the Clarity dataset
 * into Neo4j. It supports both creating new nodes and updating existing ones, making it
 * suitable for both initial setup and incremental updates.
 *
 * <h2>Import Strategy</h2>
 * The importer uses an upsert approach:
 * <ol>
 *   <li>For each QA, checks if a node already exists based on index and split flags</li>
 *   <li>If exists: Updates the existing node with new data</li>
 *   <li>If not exists: Creates a new node in the database</li>
 * </ol>
 *
 * <h2>Node Properties</h2>
 * Each imported QA node contains:
 * <ul>
 *   <li><b>index</b>: Unique identifier from the dataset (unique for test/train)</li>
 *   <li><b>test</b>: Boolean indicating test split membership</li>
 *   <li><b>valid</b>: Boolean indicating validation split membership</li>
 *   <li><b>train</b>: Boolean indicating training split membership</li>
 *   <li><b>question</b>: The extracted question text</li>
 *   <li><b>interviewQuestion</b>: Interview context question</li>
 *   <li><b>interviewAnswer</b>: Interview context answer</li>
 *   <li><b>clarityLabel</b>: Ground truth label for Clarity taxonomy</li>
 *   <li><b>evasionLabel</b>: Ground truth label for Evasion taxonomy</li>
 *   <li>Additional metadata fields from the QA model</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * Uses parallel streams for improved import performance on large datasets.
 * Each QA import operation is independent and thread-safe.
 *
 * <h2>Error Handling</h2>
 * <ul>
 *   <li>Logs detailed information for each import operation</li>
 *   <li>Logs errors for failed imports without stopping the batch</li>
 *   <li>Continues processing remaining QAs even if some imports fail</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>
 * DatasetGraphImporter importer = new DatasetGraphImporter();
 * List&lt;QA&gt; dataset = loadDataset();
 * importer.importDataset(dataset);
 * </pre>
 *
 * @see QA
 * @see DatasetReader
 * @see CleanedDataImporter
 */
public class DatasetGraphImporter {
    /**
     * Logger instance for logging information and errors.
     */
    private final Logger log = org.slf4j.LoggerFactory.getLogger(DatasetGraphImporter.class);

    /**
     * Neo4j client for database operations.
     */
    private final Neo4jClient client;

    /**
     * Constructs a DatasetGraphImporter with a Neo4j client using default credentials.
     * <p>
     * Credentials are loaded from the default location defined in the Neo4jClient.
     *
     * @throws IOException if there is an error initializing the Neo4j client
     */
    public DatasetGraphImporter() throws IOException {
        this.client = new Neo4jClient();
    }

    /**
     * Constructs a DatasetGraphImporter with a Neo4j client using credentials from the specified file.
     *
     * @param neo4jCredentialsFile path to the Neo4j credentials YAML file
     * @throws IOException if there is an error loading the credentials or initializing the Neo4j client
     */
    public DatasetGraphImporter(String neo4jCredentialsFile) throws IOException {
        Neo4jCredentials neo4jCredentials = Neo4jCredentials.load(neo4jCredentialsFile);
        this.client = new Neo4jClient(neo4jCredentials);
    }

    /**
     * Constructs a DatasetGraphImporter with a Neo4j client using the provided credentials.
     *
     * @param neo4jCredentials Neo4j database credentials object
     */
    public DatasetGraphImporter(Neo4jCredentials neo4jCredentials) {
        this.client = new Neo4jClient(neo4jCredentials);
    }

    /**
     * Imports or updates QA pairs in the Neo4j graph database.
     * <p>
     * This method implements an upsert strategy:
     * <ol>
     *   <li>For each QA, searches for an existing node by index and split flags</li>
     *   <li>If found: Updates the existing node with current data</li>
     *   <li>If not found: Creates a new node with all properties</li>
     * </ol>
     * <p>
     * Processing is performed in parallel for improved performance on large datasets.
     *
     * <h3>Node Matching Criteria</h3>
     * Existing nodes are identified by:
     * <ul>
     *   <li><b>index</b>: Unique identifier from the dataset</li>
     *   <li><b>test</b>: Test split flag</li>
     *   <li><b>valid</b>: Validation split flag</li>
     *   <li><b>train</b>: Training split flag</li>
     * </ul>
     *
     * <h3>Logging</h3>
     * <ul>
     *   <li>Info: Logs each import or update operation</li>
     *   <li>Error: Logs failures with QA index for debugging</li>
     * </ul>
     *
     * @param dataset list of QA records to be imported or updated
     */
    public void importDataset(List<QA> dataset) {
        dataset.parallelStream().forEach(qa -> {
            try {
                Map<String, Object> properties = Map.of(
                        "index", qa.getIndex(),
                        "test", qa.isTest(),
                        "valid", qa.isValid(),
                        "train", qa.isTrain()
                );
                QA existingNode = client.findNode(properties, QA.class);
                if (existingNode != null) {
                    log.info("QA pair {} already exists in Neo4j. Updating node", qa.getIndex());
                    qa.setElementId(existingNode.getElementId());
                    client.updateNode(qa);
                } else {
                    log.info("Importing QA pair {} to Neo4j", qa.getIndex());
                    client.saveNode(qa);
                }
            } catch (Exception e) {
                log.error("Error importing QA pair {} to Neo4j: {}", qa.getIndex(), e.getMessage());
            }
        });
    }
}
