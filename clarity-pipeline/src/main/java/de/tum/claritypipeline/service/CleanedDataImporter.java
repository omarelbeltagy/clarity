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
public class CleanedDataImporter {
    /**
     * Logger instance for logging information and errors.
     */
    private final Logger log = org.slf4j.LoggerFactory.getLogger(CleanedDataImporter.class);

    /**
     * Neo4j client for database operations.
     */
    private final Neo4jClient client;

    /**
     * Constructs a DatasetGraphImporter with a Neo4j client with default credentials.
     *
     * @throws IOException if there is an error initializing the Neo4j client.
     */
    public CleanedDataImporter() throws IOException {
        this.client = new Neo4jClient();
    }

    /**
     * Constructs a DatasetGraphImporter with a Neo4j client using credentials from the specified file.
     *
     * @param neo4jCredentialsFile Path to the Neo4j credentials file.
     * @throws IOException if there is an error loading the credentials or initializing the Neo4j client.
     */
    public CleanedDataImporter(String neo4jCredentialsFile) throws IOException {
        Neo4jCredentials neo4jCredentials = Neo4jCredentials.load(neo4jCredentialsFile);
        this.client = new Neo4jClient(neo4jCredentials);
    }

    /**
     * Constructs a DatasetGraphImporter with a Neo4j client using the provided credentials.
     *
     * @param neo4jCredentials Neo4j database credentials.
     */
    public CleanedDataImporter(Neo4jCredentials neo4jCredentials) {
        this.client = new Neo4jClient(neo4jCredentials);
    }

    public void importCleanedData(List<QA> dataset) {
        dataset.parallelStream().forEach(qa -> {
            try {
                Map<String, Object> properties = Map.of(
                        "index", qa.getIndex(),
                        "test", qa.isTest()
                );
                QA existingNode = client.findNode(properties, QA.class);
                if (existingNode != null) {
                    existingNode.setInterviewQuestionClean(qa.getInterviewQuestionClean());
                    existingNode.setInterviewAnswerClean(qa.getInterviewAnswerClean());
                    client.updateNode(existingNode);
                } else {
                    log.warn("Did not find node {}", qa.getIndex());
                }
            } catch (Exception e) {
                log.error("Error importing QA pair {} to Neo4j: {}", qa.getIndex(), e.getMessage());
            }
        });
    }
}
