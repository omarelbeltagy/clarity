package de.tum.clarityneo4j.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import org.neo4j.driver.*;
import org.neo4j.driver.Record;
import org.slf4j.Logger;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Neo4jExporter provides functionality to export and import Neo4j database contents as JSON,
 * as well as to clear the database.
 */
public class Neo4jExporter {
    private final Logger log = org.slf4j.LoggerFactory.getLogger(Neo4jExporter.class);

    private final Driver driver;

    private static final int RELATIONSHIPS_BATCH_SIZE = 1000;
    private static final int NODES_BATCH_SIZE = 1000;

    /**
     * Constructs a Neo4jExporter using default credentials.
     *
     * @throws IOException if default credentials cannot be loaded
     */
    public Neo4jExporter() throws IOException {
        this(Neo4jCredentials.getDefault());
    }

    /**
     * Constructs a Neo4jExporter with the specified credentials.
     *
     * @param neo4jCredentials credentials for Neo4j connection
     */
    public Neo4jExporter(Neo4jCredentials neo4jCredentials) {
        this.driver = GraphDatabase.driver(neo4jCredentials.getNeo4jUrl(),
                                           AuthTokens.basic(neo4jCredentials.getNeo4jUser(),
                                                            neo4jCredentials.getNeo4jPassword()));
    }

    /**
     * Constructs a Neo4jExporter with an existing Neo4j driver.
     *
     * @param driver Neo4j driver instance
     */
    public Neo4jExporter(Driver driver) {
        this.driver = driver;
    }

    /**
     * Exports the entire Neo4j database as JSON to the specified output file.
     * Uses the APOC procedure 'apoc.export.json.all' to stream the database contents.
     *
     * @param outputFile path to the output JSON file
     * @throws IOException if writing to the file fails
     */
    public void exportAsJson(String outputFile) throws IOException {
        try (Session session = driver.session()) {
            Result result = session.run(
                    "CALL apoc.export.json.all(null, {stream:true}) YIELD data RETURN data"
            );
            StringBuilder sb = new StringBuilder();
            while (result.hasNext()) {
                Record record = result.next();
                sb.append(record.get("data").asString());
            }
            Files.write(Paths.get(outputFile), sb.toString().getBytes());
        }
    }

    /**
     * Clears the entire Neo4j database by deleting all nodes and relationships.
     * Logs the operation.
     */
    public void clearDatabase() {
        String cypherQuery = "MATCH (n) DETACH DELETE n";
        this.driver.session().run(cypherQuery);
        log.info("Cleared the database.");
    }

    /**
     * Imports nodes and relationships from a JSON file into the Neo4j database using APOC procedures.
     * The JSON file should contain one JSON object per line, each representing either a node or a relationship.
     * Nodes are created first, and their original IDs are temporarily stored as properties.
     * Relationships are then created using these temporary IDs.
     * Finally, the temporary IDs are removed from the nodes.
     *
     * @param inputFile path to the input JSON file
     * @throws IOException if reading the file fails
     */
    public void importFromJson(String inputFile) throws IOException {
        ObjectMapper mapper = new ObjectMapper();

        log.info("Reading JSON data from file: {}", inputFile);

        Duration durationStart = Duration.ofMillis(System.currentTimeMillis());

        int lineCount = 0;
        List<JsonNode> nodeBatch = new ArrayList<>();
        List<JsonNode> relationshipBatch = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(inputFile))) {
            String line;

            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                lineCount++;
                JsonNode node = mapper.readTree(line);
                if (!node.has("properties") || node.get("properties").isNull()) {
                    ((ObjectNode) node).set("properties", mapper.createObjectNode());
                }
                if ("node".equals(node.get("type").asText())) {
                    nodeBatch.add(node);
                    Object id = node.get("id").asLong();
                    ((ObjectNode) node.get("properties")).put("tempId", id.toString());
                } else if ("relationship".equals(node.get("type").asText())) {
                    Object startId = node.get("start").get("id").asLong();
                    Object endId = node.get("end").get("id").asLong();
                    ((ObjectNode) node).put("start", startId.toString());
                    ((ObjectNode) node).put("end", endId.toString());
                    relationshipBatch.add(node);
                }
            }
        }
        Duration readDuration = Duration.ofMillis(System.currentTimeMillis()).minus(durationStart);

        log.info("Total lines read: {} in {} ms.", lineCount, readDuration.toMillis());

        log.info("Importing {} nodes with batch size {}.", nodeBatch.size(), NODES_BATCH_SIZE);

        Record nodesRecord = importNodeBatch(nodeBatch, mapper);
        Duration nodeImportDuration = Duration.ofMillis(System.currentTimeMillis()).minus(durationStart)
                                              .minus(readDuration);
        log.info("Node import completed in {} ms.", nodeImportDuration.toMillis());

        log.info("Importing {} relationships with batch size {}.", relationshipBatch.size(),
                 RELATIONSHIPS_BATCH_SIZE);

        Record relationshipsRecord = importRelationshipBatch(relationshipBatch, mapper);
        Duration relImportDuration = Duration.ofMillis(System.currentTimeMillis()).minus(durationStart)
                                             .minus(readDuration).minus(nodeImportDuration);
        log.info("Relationship import completed in {} ms.", relImportDuration.toMillis());

        log.info("Imported {} nodes and {} relationships.", nodesRecord.get("total").asInt(),
                 relationshipsRecord.get("total").asInt()); ;

        Record removeTempIdsRecord = removeTempIds();
        Duration cleanupDuration = Duration.ofMillis(System.currentTimeMillis()).minus(durationStart)
                                           .minus(readDuration).minus(nodeImportDuration).minus(relImportDuration);
        log.info("Removed temporary IDs from {} nodes in {} ms.", removeTempIdsRecord.get("nodesUpdated").asInt(),
                 cleanupDuration.toMillis());
    }

    /**
     * Imports a batch of nodes into the Neo4j database using APOC procedures.
     *
     * @param nodes  list of JSON nodes to import
     * @param mapper ObjectMapper for JSON processing
     * @return Record containing the result of the import operation
     */
    private Record importNodeBatch(List<JsonNode> nodes, ObjectMapper mapper) {
        try (Session session = driver.session()) {
            return session.executeWrite(tx -> tx.run("""
                                                             CALL apoc.periodic.iterate(
                                                                 'UNWIND $nodes as nodeData RETURN nodeData',
                                                                 '
                                                                 CALL apoc.create.node(nodeData.labels, nodeData.properties)
                                                                 YIELD node
                                                                 RETURN node
                                                                 ',
                                                                 {batchSize:""" + NODES_BATCH_SIZE + """
                                                        , parallel: true, params: {nodes: $nodes}}
                                                        )
                                                        """, Values.parameters("nodes", nodes.stream()
                                                                                             .map(n -> mapper.convertValue(
                                                                                                     n,
                                                                                                     Map.class))
                                                                                             .collect(
                                                                                                     Collectors.toList())))
                                                .single());
        }
    }

    /**
     * Imports a batch of relationships into the Neo4j database using APOC procedures.
     *
     * @param relationships list of JSON relationships to import
     * @param mapper        ObjectMapper for JSON processing
     * @return Record containing the result of the import operation
     */
    private Record importRelationshipBatch(List<JsonNode> relationships, ObjectMapper mapper) {
        try (Session session = driver.session()) {
            return session.executeWrite(tx -> tx.run("""
                                                             CALL apoc.periodic.iterate(
                                                                 'UNWIND $relationships as relData RETURN relData',
                                                                 '
                                                                 MATCH (a), (b)
                                                                 WHERE a.tempId = relData.start
                                                                     AND b.tempId = relData.end
                                                                 CALL apoc.create.relationship(a, relData.label, relData.properties, b)
                                                                 YIELD rel
                                                                 RETURN rel
                                                                 ',
                                                                 {batchSize:""" + RELATIONSHIPS_BATCH_SIZE + """
                                                            , parallel: true, params: {relationships: $relationships}}
                                                        )
                                                        """, Values.parameters("relationships", relationships.stream()
                                                                                                             .map(n -> mapper.convertValue(
                                                                                                                     n,
                                                                                                                     Map.class))
                                                                                                             .collect(
                                                                                                                     Collectors.toList())))
                                                .single());
        }
    }

    /**
     * Removes temporary IDs from all nodes in the Neo4j database.
     *
     * @return Record containing the result of the cleanup operation
     */
    private Record removeTempIds() {
        return driver.session().run("""
                                                           MATCH (n)
                                                           REMOVE n.tempId
                                                           RETURN count(n) AS nodesUpdated
                                            """).single();
    }
}
