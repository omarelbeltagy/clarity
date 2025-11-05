package de.tum.clarityneo4j.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import org.neo4j.driver.*;
import org.neo4j.driver.Record;
import org.slf4j.Logger;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Neo4jExporter provides functionality to export and import Neo4j database contents as JSON,
 * as well as to clear the database.
 */
public class Neo4jExporter {
    private final Logger log = org.slf4j.LoggerFactory.getLogger(Neo4jExporter.class);

    private final Driver driver;

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
     * Imports nodes and relationships from a JSON file into the Neo4j database.
     * The JSON file should contain one JSON object per line, each representing either a node or a relationship.
     * Nodes are created first, and their original IDs are mapped to Neo4j element IDs.
     * Relationships are then created using these mapped IDs.
     * Logs the number of successful and failed imports for nodes and relationships.
     *
     * @param inputFile path to the input JSON file
     * @throws IOException if reading the file fails
     */
    public void importFromJson(String inputFile) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        int nodeCountSuccess = 0;
        int relCountSuccess = 0;
        int nodeCountFail = 0;
        int relCountFail = 0;

        try (BufferedReader reader = new BufferedReader(new FileReader(inputFile));
             Session session = driver.session()) {

            Map<Long, String> idMapping = new HashMap<>();

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;

                JsonNode node = mapper.readTree(line);

                String type = node.get("type").asText();
                if ("node".equals(type)) {
                    String labels = "";
                    if (node.has("labels")) {
                        StringBuilder sb = new StringBuilder();
                        for (JsonNode l : node.get("labels")) {
                            sb.append(":").append(l.asText());
                        }
                        labels = sb.toString();
                    }
                    JsonNode props = node.get("properties");
                    Stream<Record> records = session.run("CREATE (n" + labels + " $props) RETURN n",
                                                         Values.parameters("props", mapper.convertValue(props,
                                                                                                        java.util.Map.class)))
                                                    .stream();

                    Record record = records.findFirst().orElse(null);
                    if (record != null) {
                        long originalId = node.get("id").asLong();
                        String elementId = record.get("n").asNode().elementId();
                        idMapping.put(originalId, elementId);
                        log.debug("Import node with labels {} and elementId {}", labels, elementId);
                        nodeCountSuccess++;
                    } else {
                        log.warn("Failed to create node with labels {} and properties {}", labels, props);
                        nodeCountFail++;
                    }
                } else if ("relationship".equals(type)) {
                    long startOriginalId = node.get("start").get("id").asLong();
                    long endOriginalId = node.get("end").get("id").asLong();
                    String startElementId = idMapping.get(startOriginalId);
                    String endElementId = idMapping.get(endOriginalId);

                    if (startElementId == null || endElementId == null) {
                        log.warn("Skipping relationship import due to missing nodes: startId={}, endId={}",
                                 startOriginalId, endOriginalId);
                        nodeCountFail++;
                        continue;
                    }

                    String relType = node.get("label").asText();
                    JsonNode props = node.get("properties");

                    if (props == null) {
                        props = mapper.createObjectNode();
                    }

                    Stream<Record> records = session.run(
                            String.format("""
                                                  MATCH (a) WHERE elementId(a)='%s'
                                                  MATCH (b) WHERE elementId(b)='%s'
                                                  CREATE (a)-[r:%s $props]->(b)
                                                  RETURN r
                                                  """,
                                          startElementId,
                                          endElementId,
                                          relType),
                            Values.parameters(
                                    "props", mapper.convertValue(props, java.util.Map.class)
                            )
                    ).stream();
                    Record record = records.findFirst().orElse(null);
                    if (record != null) {
                        String relElementId = record.get("r").asRelationship().elementId();
                        log.debug("Imported relationship of type {} with elementId {}", relType, relElementId);
                        relCountSuccess++;
                    } else {
                        log.warn("Failed to create relationship of type {} between {} and {} with properties {}",
                                 relType, startElementId, endElementId, props);
                        relCountFail++;
                    }
                }
            }

            log.info(
                    "Import completed: {} nodes succeeded, {} nodes failed, {} relationships succeeded, {} "
                            + "relationships failed.",
                    nodeCountSuccess, nodeCountFail, relCountSuccess, relCountFail);
        }
    }
}
