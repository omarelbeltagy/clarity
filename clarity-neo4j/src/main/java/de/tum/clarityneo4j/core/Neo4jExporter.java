package de.tum.clarityneo4j.core;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import de.tum.clarityneo4j.model.Neo4jExporterConfig;
import org.neo4j.driver.*;
import org.neo4j.driver.Record;
import org.slf4j.Logger;

import java.io.*;
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

    private final Driver driver;
    private final Neo4jExporterConfig neo4jExporterConfig;
    private final Logger log = org.slf4j.LoggerFactory.getLogger(Neo4jExporter.class);

    public Neo4jExporter() throws IOException {
        this(Neo4jExporterConfig.getDefault());
    }

    public Neo4jExporter(Neo4jExporterConfig neo4jExporterConfig) {
        this.neo4jExporterConfig = neo4jExporterConfig;
        this.driver = GraphDatabase.driver(neo4jExporterConfig.getNeo4jCredentials().getNeo4jUrl(),
                                           AuthTokens.basic(neo4jExporterConfig.getNeo4jCredentials().getNeo4jUser(),
                                                            neo4jExporterConfig.getNeo4jCredentials()
                                                                               .getNeo4jPassword()));
    }

    /**
     * Exports the entire Neo4j database as JSON to the specified output file.
     * Uses the APOC procedure 'apoc.export.json.all' to stream the database contents.
     *
     * @param outputFile path to the output JSON file
     * @throws IOException if writing to the file fails
     */
    public void exportAsJson(String outputFile) throws IOException {
        exportAsJson(outputFile, true);
    }

    public void exportAsJson(String outputFile, boolean exportEmbeddings) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {

            try (Session session = driver.session()) {
                long totalNodes = session.run("MATCH (n) RETURN count(n) as count")
                                         .single().get("count").asLong();

                log.info("Starting export of {} nodes.", totalNodes);
                Duration startDurationExport = Duration.ofMillis(System.currentTimeMillis());

                exportNodesInBatches(session, writer, totalNodes, exportEmbeddings);
                Duration endDurationExport = Duration.ofMillis(System.currentTimeMillis()).minus(startDurationExport);
                log.info("Node export completed in {} ms.", endDurationExport.toMillis());

                long totalRels = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                                        .single().get("count").asLong();

                log.info("Starting export of {} relationships.", totalRels);
                startDurationExport = Duration.ofMillis(System.currentTimeMillis());

                exportRelationshipsInBatches(session, writer, totalRels);

                endDurationExport = Duration.ofMillis(System.currentTimeMillis()).minus(startDurationExport);
                log.info("Relationship export completed in {} ms.", endDurationExport.toMillis());

                log.info("Export completed in {} ms.",
                         Duration.ofMillis(System.currentTimeMillis()).minus(startDurationExport).toMillis());

            }
        }
    }

    private void exportNodesInBatches(
            Session session, BufferedWriter writer,
            long totalNodes, boolean exportEmbeddings
    ) throws IOException {
        long processed = 0;

        while (processed < totalNodes) {
            Result result = session.run(
                    """
                            MATCH (n)
                            WITH n
                            SKIP $skip LIMIT $limit
                            CALL apoc.export.json.data([n], [], null, {stream: true})
                            YIELD data
                            RETURN data
                            """,
                    Map.of("skip", processed, "limit",
                           neo4jExporterConfig.getBatchConfig().getExportConfig().getNodeBatchSize())
            );

            ObjectMapper mapper = new ObjectMapper();
            while (result.hasNext()) {
                String data = result.next().get("data").asString().trim();
                if (!data.isEmpty()) {
                    if (!exportEmbeddings) {
                        data = removeEmbeddingProperties(data, mapper);
                    }
                    writer.write(data);
                    writer.newLine();
                }
            }

            processed += neo4jExporterConfig.getBatchConfig().getExportConfig().getNodeBatchSize();
            if (processed > totalNodes) {
                processed = totalNodes;
            }
            log.info("Nodes: {} / {}", processed, totalNodes);
        }
    }

    private String removeEmbeddingProperties(String jsonLine, ObjectMapper mapper) {
        try {
            JsonNode obj = mapper.readTree(jsonLine);

            if (obj.has("properties") && obj.get("properties").isObject()) {
                ObjectNode properties = (ObjectNode) obj.get("properties");

                properties.remove("questionAnswerEmbedding");
                properties.remove("questionEmbedding");
                properties.remove("answerEmbedding");
            }

            return mapper.writeValueAsString(obj);
        } catch (Exception e) {
            log.warn("Failed to parse/modify JSON line, returning original: {}", e.getMessage());
            return jsonLine;
        }
    }

    private void exportRelationshipsInBatches(
            Session session, BufferedWriter writer,
            long totalRels
    ) throws IOException {
        long processed = 0;

        while (processed < totalRels) {
            Result result = session.run(
                    "MATCH ()-[r]->() " +
                            "WITH r " +
                            "SKIP $skip LIMIT $limit " +
                            "CALL apoc.export.json.data([], [r], null, {stream: true}) " +
                            "YIELD data " +
                            "RETURN data",
                    Map.of("skip", processed, "limit",
                           neo4jExporterConfig.getBatchConfig().getExportConfig().getRelationshipBatchSize())
            );

            ObjectMapper mapper = new ObjectMapper();
            while (result.hasNext()) {
                String data = result.next().get("data").asString().trim();
                if (!data.isEmpty()) {
                    data = extractStartAndEndNodeIds(data, mapper);
                    writer.write(data);
                    writer.newLine();
                }
            }

            processed += neo4jExporterConfig.getBatchConfig().getExportConfig().getRelationshipBatchSize();
            if (processed > totalRels) {
                processed = totalRels;
            }
            log.info("Relationships: {} / {}", processed, totalRels);
        }
    }

    private String extractStartAndEndNodeIds(String jsonLine, ObjectMapper mapper) {
        try {
            JsonNode obj = mapper.readTree(jsonLine);

            if (obj.has("start") && obj.get("start").isObject()) {
                ObjectNode startNode = (ObjectNode) obj.get("start");
                long startId = startNode.get("id").asLong();
                ((ObjectNode) obj).remove("start");
                ((ObjectNode) obj).put("start", startId);
            }

            if (obj.has("end") && obj.get("end").isObject()) {
                ObjectNode endNode = (ObjectNode) obj.get("end");
                long endId = endNode.get("id").asLong();
                ((ObjectNode) obj).remove("end");
                ((ObjectNode) obj).put("end", endId);
            }

            return mapper.writeValueAsString(obj);
        } catch (Exception e) {
            log.warn("Failed to parse/modify JSON line, returning original: {}", e.getMessage());
            return jsonLine;
        }
    }

    /**
     * Clears the entire Neo4j database by deleting all nodes and relationships.
     * Logs the operation.
     */
    public void clearDatabase() {
        int deletedCount;
        int totalDeleted = 0;

        log.info("Starting database cleanup in batches of {}...",
                 neo4jExporterConfig.getBatchConfig().getDeleteBatchSize());

        try (Session session = driver.session()) {
            do {
                String cypherQuery = String.format(
                        "MATCH (n) " +
                                "WITH n LIMIT %d " +
                                "DETACH DELETE n " +
                                "RETURN count(n) as deleted",
                        neo4jExporterConfig.getBatchConfig().getDeleteBatchSize()
                );

                Result result = session.run(cypherQuery);
                deletedCount = result.single().get("deleted").asInt();
                totalDeleted += deletedCount;

                log.debug("Deleted {} nodes (total: {})", deletedCount, totalDeleted);

            } while (deletedCount > 0);

            log.info("Database cleared. Total nodes deleted: {}", totalDeleted);
        }
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
        log.info("Starting import from JSON file: {}", inputFile);

        ObjectMapper mapper = new ObjectMapper();
        Duration durationStart = Duration.ofMillis(System.currentTimeMillis());

        int totalNodes = 0;
        int totalRelationships = 0;

        try (BufferedReader reader = new BufferedReader(new FileReader(inputFile), 1024 * 1024)) {
            String line;
            List<JsonNode> nodeBatch = new ArrayList<>();

            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;

                JsonNode node = mapper.readTree(line);
                if (!node.has("properties") || node.get("properties").isNull()) {
                    ((ObjectNode) node).set("properties", mapper.createObjectNode());
                }

                if ("node".equals(node.get("type").asText())) {
                    Object id = node.get("id").asLong();
                    ((ObjectNode) node.get("properties")).put("tempId", id.toString());
                    nodeBatch.add(node);

                    if (nodeBatch.size() >= neo4jExporterConfig.getBatchConfig().getImportConfig()
                                                               .getReadBatchSize()) {
                        Record record = importNodeBatch(nodeBatch, mapper);
                        totalNodes += record.get("total").asInt();
                        nodeBatch.clear();
                        log.info("Imported {} nodes so far...", totalNodes);
                    }
                }
            }

            if (!nodeBatch.isEmpty()) {
                Record record = importNodeBatch(nodeBatch, mapper);
                totalNodes += record.get("total").asInt();
            }
        }

        log.info("Node import completed. Total nodes: {}", totalNodes);

        log.info("Creating lookup map for element IDs on temporary IDs...");
        Map<String, String> tempIdToElementIdMap = driver.session().run(
                                                                 """
                                                                         MATCH (n)
                                                                         WHERE n.tempId IS NOT NULL
                                                                         RETURN n.tempId AS tempId, elementId(n) AS elementId
                                                                         """
                                                         ).list(r -> Map.entry(r.get("tempId").asString(),
                                                                               String.valueOf(r.get("elementId").asString())))
                                                         .stream().collect(
                        Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        log.info("Lookup map created with {} entries.", tempIdToElementIdMap.size());

        try (BufferedReader reader = new BufferedReader(new FileReader(inputFile), 1024 * 1024)) {
            String line;
            List<JsonNode> relationshipBatch = new ArrayList<>();

            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;

                JsonNode node = mapper.readTree(line);
                if (!node.has("properties") || node.get("properties").isNull()) {
                    ((ObjectNode) node).set("properties", mapper.createObjectNode());
                }

                if ("relationship".equals(node.get("type").asText())) {
                    String startId;
                    if (node.get("start").isObject()) {
                        startId = node.get("start").get("id").asText();
                    } else {
                        startId = node.get("start").asText();
                    }
                    String endId;
                    if (node.get("end").isObject()) {
                        endId = node.get("end").get("id").asText();
                    } else {
                        endId = node.get("end").asText();
                    }

                    ((ObjectNode) node).put("startElementId", tempIdToElementIdMap.get(startId));
                    ((ObjectNode) node).put("endElementId", tempIdToElementIdMap.get(endId));
                    relationshipBatch.add(node);

                    if (relationshipBatch.size() >= neo4jExporterConfig.getBatchConfig()
                                                                       .getImportConfig()
                                                                       .getReadBatchSize()) {
                        Record record = importRelationshipBatch(relationshipBatch, mapper);
                        totalRelationships += record.get("total").asInt();
                        relationshipBatch.clear();
                        log.info("Imported {} relationships so far...", totalRelationships);
                    }
                }
            }

            if (!relationshipBatch.isEmpty()) {
                Record record = importRelationshipBatch(relationshipBatch, mapper);
                totalRelationships += record.get("total").asInt();
            }
        }

        log.info("Relationship import completed. Total relationships: {}", totalRelationships);

        Record removeTempIdsRecord = removeTempIds();
        log.info("Removed temporary IDs from {} nodes.", removeTempIdsRecord.get("nodesUpdated").asInt());

        Duration durationEnd = Duration.ofMillis(System.currentTimeMillis()).minus(durationStart);
        log.info("Import completed in {} ms.", durationEnd.toMillis());
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
                                                                 {batchSize:""" + neo4jExporterConfig.getBatchConfig()
                                                                                                     .getImportConfig()
                                                                                                     .getNodeBatchSize()
                                                             + """
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
                                                                 WHERE elementId(a) = relData.startElementId
                                                                   AND elementId(b) = relData.endElementId
                                                                 CALL apoc.create.relationship(a, relData.label, relData.properties, b)
                                                                 YIELD rel
                                                                 RETURN rel
                                                                 ',
                                                                 {batchSize:""" + neo4jExporterConfig.getBatchConfig()
                                                                                                     .getImportConfig()
                                                                                                     .getRelationshipBatchSize()
                                                             + """
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