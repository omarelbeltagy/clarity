package de.tum.claritypipeline.service;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.claritypipeline.model.core.QA;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Service for importing QA dataset into a Neo4j graph database.
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
     * Constructs a DatasetGraphImporter with a Neo4j client with default credentials.
     *
     * @throws IOException if there is an error initializing the Neo4j client.
     */
    public DatasetGraphImporter() throws IOException {
        this.client = new Neo4jClient();
    }

    /**
     * Constructs a DatasetGraphImporter with a Neo4j client using credentials from the specified file.
     *
     * @param neo4jCredentialsFile Path to the Neo4j credentials file.
     * @throws IOException if there is an error loading the credentials or initializing the Neo4j client.
     */
    public DatasetGraphImporter(String neo4jCredentialsFile) throws IOException {
        Neo4jCredentials neo4jCredentials = Neo4jCredentials.load(neo4jCredentialsFile);
        this.client = new Neo4jClient(neo4jCredentials);
    }

    /**
     * Constructs a DatasetGraphImporter with a Neo4j client using the provided credentials.
     *
     * @param neo4jCredentials Neo4j database credentials.
     */
    public DatasetGraphImporter(Neo4jCredentials neo4jCredentials) {
        this.client = new Neo4jClient(neo4jCredentials);
    }

    /**
     * Imports the given dataset into the Neo4j graph database.
     * <p>
     * The method checks for existing QA nodes based on the "index" property.
     * If a node with the same index exists, it updates the node.
     * Otherwise, it creates a new node.
     *
     * @param dataset List of QA records to be imported.
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
