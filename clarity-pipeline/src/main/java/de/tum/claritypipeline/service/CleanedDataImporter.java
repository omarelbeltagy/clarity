package de.tum.claritypipeline.service;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.claritypipeline.model.core.QA;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Service for importing cleaned QA data into an existing Neo4j graph database.
 * <p>
 * This service is designed to update already imported QA nodes with their cleaned versions
 * of interview questions and answers. It operates by:
 * <ol>
 *   <li>Finding existing QA nodes in Neo4j by their index and test flag</li>
 *   <li>Updating the cleaned text fields (interviewQuestionClean, interviewAnswerClean)</li>
 *   <li>Preserving all other node properties and relationships</li>
 * </ol>
 *
 * <h2>Use Case</h2>
 * This service is typically used after initial dataset import when cleaned/preprocessed
 * versions of the interview data become available. It allows enriching existing QA nodes
 * without reimporting the entire dataset.
 *
 * <h2>Thread Safety</h2>
 * Uses parallel streams for improved performance when processing large datasets.
 * Each QA update is handled independently.
 *
 * <h2>Error Handling</h2>
 * <ul>
 *   <li>Logs warnings when matching nodes are not found</li>
 *   <li>Logs errors for individual import failures without stopping the batch</li>
 *   <li>Continues processing remaining QAs even if some updates fail</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>
 * CleanedDataImporter importer = new CleanedDataImporter("neo4j-credentials.yaml");
 * List&lt;QA&gt; cleanedData = loadCleanedDataset();
 * importer.importCleanedData(cleanedData);
 * </pre>
 *
 * @see QA
 * @see DatasetGraphImporter
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
     * Constructs a CleanedDataImporter with a Neo4j client using default credentials.
     * <p>
     * Credentials are loaded from the default location defined in the Neo4jClient.
     *
     * @throws IOException if there is an error initializing the Neo4j client
     */
    public CleanedDataImporter() throws IOException {
        this.client = new Neo4jClient();
    }

    /**
     * Constructs a CleanedDataImporter with a Neo4j client using credentials from the specified file.
     *
     * @param neo4jCredentialsFile path to the Neo4j credentials YAML file
     * @throws IOException if there is an error loading the credentials or initializing the Neo4j client
     */
    public CleanedDataImporter(String neo4jCredentialsFile) throws IOException {
        Neo4jCredentials neo4jCredentials = Neo4jCredentials.load(neo4jCredentialsFile);
        this.client = new Neo4jClient(neo4jCredentials);
    }

    /**
     * Constructs a CleanedDataImporter with a Neo4j client using the provided credentials.
     *
     * @param neo4jCredentials Neo4j database credentials object
     */
    public CleanedDataImporter(Neo4jCredentials neo4jCredentials) {
        this.client = new Neo4jClient(neo4jCredentials);
    }

    /**
     * Imports cleaned interview data into existing QA nodes in Neo4j.
     * <p>
     * For each QA in the dataset:
     * <ol>
     *   <li>Searches for an existing node matching the index and test flag</li>
     *   <li>If found: Updates the cleaned question and answer fields</li>
     *   <li>If not found: Logs a warning (node should exist from initial import)</li>
     * </ol>
     * <p>
     * Processing is performed in parallel for improved performance on large datasets.
     *
     * <h3>Node Matching Criteria</h3>
     * Nodes are matched using:
     * <ul>
     *   <li><b>index</b>: Unique identifier from the original dataset</li>
     *   <li><b>test</b>: Boolean flag indicating test split membership</li>
     * </ul>
     *
     * <h3>Updated Properties</h3>
     * <ul>
     *   <li><b>interviewQuestionClean</b>: Cleaned version of the interview question</li>
     *   <li><b>interviewAnswerClean</b>: Cleaned version of the interview answer</li>
     * </ul>
     *
     * @param dataset list of QA objects containing cleaned interview data
     */
    public void importCleanedData(List<QA> dataset) {
        dataset.parallelStream().forEach(qa -> {
            try {
                Map<String, Object> properties = Map.of(
                        "question", qa.getQuestion(),
                        "test", qa.isTest(),
                        "valid", qa.isValid(),
                        "train", qa.isTrain()
                );
                List<QA> existingNodes = client.findNodes(properties, QA.class);
                if(existingNodes.isEmpty()) {
                    properties = Map.of(
                            "question", qa.getQuestion() + " ",
                            "test", qa.isTest(),
                            "valid", qa.isValid(),
                            "train", qa.isTrain()
                    );
                    existingNodes = client.findNodes(properties, QA.class);
                    if(existingNodes.isEmpty()) {
                        properties = Map.of(
                                "question", " " + qa.getQuestion(),
                                "test", qa.isTest(),
                                "valid", qa.isValid(),
                                "train", qa.isTrain()
                        );
                        existingNodes = client.findNodes(properties, QA.class);
                        if(existingNodes.isEmpty()) {
                            properties = Map.of(
                                    "question", qa.getQuestion().substring(qa.getQuestion().length() / 4, qa.getQuestion().length() - qa.getQuestion().length() / 4),
                                    "test", qa.isTest(),
                                    "valid", qa.isValid(),
                                    "train", qa.isTrain()
                            );
                            String query = """
                                    MATCH(n:QA)
                                    WHERE n.question CONTAINS $question AND n.test = $test AND n.valid = $valid AND n.train = $train
                                    RETURN n
                                    """;
                            existingNodes = client.executeQuery(query, properties, QA.class);
                            if(existingNodes.isEmpty()) {
                                properties = Map.of(
                                        "question", qa.getQuestion().substring(0, qa.getQuestion().length() / 4),
                                        "test", qa.isTest(),
                                        "valid", qa.isValid(),
                                        "train", qa.isTrain()
                                );
                                query = """
                                    MATCH(n:QA)
                                    WHERE n.question CONTAINS $question AND n.test = $test AND n.valid = $valid AND n.train = $train
                                    RETURN n
                                    """;
                                existingNodes = client.executeQuery(query, properties, QA.class);
                                if(existingNodes.isEmpty()) {
                                    properties = Map.of(
                                            "question", qa.getQuestion().substring(0, qa.getQuestion().length() / 12),
                                            "test", qa.isTest(),
                                            "valid", qa.isValid(),
                                            "train", qa.isTrain()
                                    );
                                    query = """
                                    MATCH(n:QA)
                                    WHERE n.question CONTAINS $question AND n.test = $test AND n.valid = $valid AND n.train = $train
                                    RETURN n
                                    """;
                                    existingNodes = client.executeQuery(query, properties, QA.class);
                                }
                            }
                        }
                    }
                }
                if(existingNodes.size() > 1) {
                    log.warn("Found multiple nodes for question {} and test flag {}, expected only one. Skipping update.", qa.getQuestion(), qa.isTest());
                    return;
                }
                QA existingNode = existingNodes.isEmpty() ? null : existingNodes.getFirst();
                if (existingNode != null) {
                    if(qa.getQuestionClean() == null || qa.getContextClean() == null) {
                        log.warn("Cleaned question or context is null for QA index {}, skipping update.", qa.getIndex());
                        return;
                    }
                    existingNode.setQuestionClean(qa.getQuestionClean());
                    existingNode.setContextClean(qa.getContextClean());
                    client.updateNode(existingNode);
                } else {
                    log.warn("Did not find node {}", qa.getQuestion());
                }
            } catch (Exception e) {
                log.error("Error importing QA pair {} to Neo4j: {}", qa.getIndex(), e.getMessage());
            }
        });
    }
}
