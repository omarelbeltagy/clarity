package de.tum.clarityneo4j.core;

import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.clarityneo4j.model.Neo4jEmbeddingSearchResult;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Record;
import org.neo4j.driver.internal.value.NodeValue;
import org.neo4j.driver.internal.value.RelationshipValue;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * Neo4jClient provides methods to interact with a Neo4j database,
 * including saving, retrieving, updating, and deleting nodes and relationships.
 */
public class Neo4jClient {
    private final Logger log = org.slf4j.LoggerFactory.getLogger(Neo4jClient.class);
    private Driver driver;

    public Neo4jClient() throws IOException {
        this(Neo4jCredentials.getDefault());
    }

    public Neo4jClient(Neo4jCredentials neo4jCredentials) {
        try {
            this.driver = GraphDatabase.driver(
                    neo4jCredentials.getNeo4jUrl(),
                    AuthTokens.basic(
                            neo4jCredentials.getNeo4jUser(),
                            neo4jCredentials.getNeo4jPassword()
                    )
            );
        } catch (Exception e) {
            log.error("Error initializing Neo4j driver: {}", e.getMessage());
        }
    }

    public Neo4jClient(Driver driver) {
        this.driver = driver;
    }

    /**
     * Saves a node to the Neo4j database using parameterized query.
     */
    public void saveNode(Neo4jNode node) {
        try {
            String alias = "n";
            String cypherQuery = String.format("""
                                                       CREATE (%s:%s $props)
                                                       RETURN %s
                                                       """, alias, node.getLabel(), alias);

            Map<String, Object> params = Map.of("props", node.toPropertiesMap());
            Stream<Record> records = getRecords(cypherQuery, params);

            records.findFirst().ifPresent(record -> {
                NodeValue n = (NodeValue) record.get(alias);
                String elementId = n.asNode().elementId();
                if (elementId == null || elementId.isEmpty()) {
                    log.error("Failed to retrieve elementId after saving node: {}", node.getLabel());
                }
                log.info("Saved node: {} with elementId: {}", node.getLabel(), elementId);
                node.setElementId(elementId);
            });
        } catch (Exception e) {
            log.error("Unexpected error while saving node with label: {}", node.getLabel(), e);
        }
    }

    /**
     * Retrieves a node by its element ID using parameterized query.
     */
    public <T extends Neo4jNode> T getNode(String elementId, Class<T> clazz) {
        try {
            String alias = "n";
            String cypherQuery = String.format("""
                                                       MATCH (%s)
                                                       WHERE elementId(n) = $elementId
                                                       RETURN %s
                                                       """, alias, alias);

            Map<String, Object> params = Map.of("elementId", elementId);
            Stream<Record> records = getRecords(cypherQuery, params);

            return records.map(record -> {
                NodeValue nodeValue = (NodeValue) record.get(alias);
                return Neo4jNode.fromNodeValue(nodeValue, clazz);
            }).findFirst().orElse(null);
        } catch (Exception e) {
            log.error("Error finding node by element ID: {}", elementId, e);
            return null;
        }
    }

    /**
     * Finds a relationship by its properties using parameterized query.
     */
    public <T extends Neo4jRelation> T findRelation(Map<String, String> properties, Class<T> clazz) {
        if (properties == null || properties.isEmpty()) {
            return null;
        }

        try {
            StringBuilder queryBuilder = new StringBuilder("MATCH ()-[r]->() WHERE ");
            Map<String, Object> params = new HashMap<>();
            int paramIndex = 0;

            for (Map.Entry<String, String> entry : properties.entrySet()) {
                if (entry.getValue() != null && !entry.getValue().isEmpty()) {
                    String paramName = "param" + paramIndex++;
                    queryBuilder.append(String.format("r.%s = $%s AND ", entry.getKey(), paramName));
                    params.put(paramName, entry.getValue());
                }
            }

            if (params.isEmpty()) {
                return null;
            }

            queryBuilder.setLength(queryBuilder.length() - 5); // Remove " AND "
            queryBuilder.append(" RETURN r");

            Stream<Record> records = getRecords(queryBuilder.toString(), params);
            return records.map(record -> {
                RelationshipValue relationValue = (RelationshipValue) record.get("r");
                return Neo4jRelation.fromRelationValue(relationValue, clazz);
            }).findFirst().orElse(null);
        } catch (Exception e) {
            log.error("Error finding relation by properties: {}", properties, e);
            return null;
        }
    }

    /**
     * Finds a relationship between two nodes using parameterized query.
     */
    public <T extends Neo4jRelation> T findRelation(String srcId, String dstId, Class<T> clazz) {
        try {
            String cypherQuery = """
                    MATCH (src)-[r]->(dst)
                    WHERE elementId(src) = $srcId AND elementId(dst) = $dstId
                    RETURN r
                    """;

            Map<String, Object> params = Map.of(
                    "srcId", srcId,
                    "dstId", dstId
            );

            Stream<Record> records = getRecords(cypherQuery, params);
            return records.map(record -> {
                RelationshipValue relationValue = (RelationshipValue) record.get("r");
                return Neo4jRelation.fromRelationValue(relationValue, clazz);
            }).findFirst().orElse(null);
        } catch (Exception e) {
            log.info("Relation of type {} between {} and {} not found.",
                     Neo4jRelation.getType(clazz), srcId, dstId);
            return null;
        }
    }

    /**
     * Finds a single node by its properties using parameterized query.
     */
    public <T extends Neo4jNode> T findNode(Map<String, Object> properties, Class<T> clazz) {
        List<T> nodes = findNodes(properties, clazz);
        if (nodes == null || nodes.isEmpty()) {
            return null;
        }
        return nodes.getFirst();
    }

    /**
     * Finds all nodes matching the given properties using parameterized query.
     */
    public <T extends Neo4jNode> List<T> findNodes(Map<String, Object> properties, Class<T> clazz) {
        String label = Neo4jNode.getLabel(clazz);
        StringBuilder queryBuilder = new StringBuilder("MATCH (n:" + label + ")");
        Map<String, Object> params = new HashMap<>();

        if (properties != null && !properties.isEmpty()) {
            queryBuilder.append(" WHERE ");
            int paramIndex = 0;

            for (Map.Entry<String, Object> entry : properties.entrySet()) {
                Object value = entry.getValue();
                if (value != null) {
                    String paramName = "param" + paramIndex++;

                    if (value instanceof Collection<?>) {
                        queryBuilder.append(String.format("n.%s IN $%s AND ", entry.getKey(), paramName));
                        params.put(paramName, value);
                    } else {
                        queryBuilder.append(String.format("n.%s = $%s AND ", entry.getKey(), paramName));
                        params.put(paramName, value);
                    }
                }
            }

            if (!params.isEmpty()) {
                queryBuilder.setLength(queryBuilder.length() - 5); // Remove " AND "
            } else {
                // If no valid properties, remove WHERE clause
                queryBuilder = new StringBuilder("MATCH (n:" + label + ")");
            }
        }

        queryBuilder.append(" RETURN n");

        try {
            Stream<Record> records = getRecords(queryBuilder.toString(), params);
            List<T> nodes = new ArrayList<>();
            records.forEach(record -> {
                NodeValue nodeValue = (NodeValue) record.get("n");
                T node = Neo4jNode.fromNodeValue(nodeValue, clazz);
                nodes.add(node);
            });
            return nodes;
        } catch (Exception e) {
            log.error("Error finding node with label {} by properties: {}", label, properties, e);
            return null;
        }
    }

    /**
     * Finds a node by properties or creates it if not found.
     */
    public <T extends Neo4jNode> T findOrCreateNode(
            Map<String, Object> search,
            Class<T> type,
            Supplier<T> creator
    ) {
        try {
            T existing = findNode(search, type);
            if (existing != null) {
                log.info("{} {} already exists", Neo4jNode.getLabel(type), search);
                return existing;
            }
            T node = creator.get();
            saveNode(node);
            return node;
        } catch (Exception e) {
            log.error("Error in findOrCreate for node with label {} and properties {}",
                      Neo4jNode.getLabel(type), search, e);
            throw new RuntimeException(e);
        }
    }

    /**
     * Creates a relationship between two nodes using parameterized query.
     */
    public <T extends Neo4jRelation> void createRelation(T relation) {
        String alias = "r";
        try {
            // Build relationship pattern based on direction
            String relPattern = buildRelationshipPattern(relation, alias);

            String cypherQuery = String.format("""
                                                       MATCH (src), (dst)
                                                       WHERE elementId(src) = $srcId
                                                       AND elementId(dst) = $dstId
                                                       CREATE (src)%s(dst)
                                                       RETURN src, dst, %s
                                                       """,
                                               relPattern,
                                               alias
            );

            Map<String, Object> params = new HashMap<>();
            params.put("srcId", relation.getStartNodeId());
            params.put("dstId", relation.getEndNodeId());
            params.put(alias + "props", relation.toPropertiesMap());

            Stream<Record> records = getRecords(cypherQuery, params);

            records.findFirst().ifPresent(record -> {
                RelationshipValue rv = (RelationshipValue) record.get(alias);
                relation.setElementId(rv.asRelationship().elementId());
                log.info("Created relationship of type {} between {} and {} with elementId: {}",
                         relation.getType(), relation.getStartNodeId(), relation.getEndNodeId(),
                         rv.asRelationship().elementId());
            });
        } catch (Exception e) {
            log.error("Error creating relationship from {} to {}: {}",
                      relation.getStartNodeId(), relation.getEndNodeId(), e.getMessage());
        }
    }

    /**
     * Builds a safe relationship pattern with parameter placeholder.
     */
    private String buildRelationshipPattern(Neo4jRelation relation, String alias) {
        String type = relation.getType();
        String propsParam = "$" + alias + "props";

        return switch (relation.getDirection()) {
            case OUTGOING -> String.format("-[%s:%s %s]->", alias, type, propsParam);
            case INCOMING -> String.format("<-[%s:%s %s]-", alias, type, propsParam);
            case UNDIRECTED -> String.format("-[%s:%s %s]-", alias, type, propsParam);
        };
    }

    /**
     * Updates an existing relationship's properties using parameterized query.
     */
    public <T extends Neo4jRelation> void updateRelation(T relation) {
        try {
            String cypherQuery = """
                    MATCH ()-[r]->()
                    WHERE elementId(r) = $elementId
                    SET r += $props
                    RETURN r
                    """;

            Map<String, Object> params = Map.of(
                    "elementId", relation.getElementId(),
                    "props", relation.toPropertiesMap()
            );

            Stream<Record> records = getRecords(cypherQuery, params);
            records.forEach(record ->
                                    log.info("Updated relationship with element ID: {}", relation.getElementId()));
        } catch (Exception e) {
            log.error("Error updating relationship with element ID: {}", relation.getElementId(), e);
        }
    }

    /**
     * Deletes a relationship by its element ID using parameterized query.
     */
    public void deleteRelation(String elementId) {
        try {
            String cypherQuery = """
                    MATCH ()-[r]->()
                    WHERE elementId(r) = $elementId
                    DELETE r
                    """;

            Map<String, Object> params = Map.of("elementId", elementId);
            Stream<Record> records = getRecords(cypherQuery, params);
            records.forEach(record -> log.debug("Deleted relation with element ID: {}", elementId));
        } catch (Exception e) {
            log.error("Error deleting relation with element ID: {}", elementId, e);
        }
    }

    /**
     * Deletes a relationship between two nodes.
     */
    public <T extends Neo4jRelation> void deleteRelation(
            String srcElementId,
            String dstElementId,
            Class<T> clazz
    ) {
        T relationship = findRelation(srcElementId, dstElementId, clazz);
        if (relationship == null) {
            log.warn("No relation of type {} found between {} and {}, nothing to delete.",
                     Neo4jRelation.getType(clazz), srcElementId, dstElementId);
            return;
        }
        deleteRelation(relationship.getElementId());
    }

    /**
     * Updates an existing node's properties using parameterized query.
     */
    public <T extends Neo4jNode> void updateNode(T neo4jNode) {
        try {
            String cypherQuery = """
                    MATCH (c)
                    WHERE elementId(c) = $elementId
                    SET c = $props
                    RETURN c
                    """;

            Map<String, Object> params = Map.of(
                    "elementId", neo4jNode.getElementId(),
                    "props", neo4jNode.toPropertiesMap()
            );

            Stream<Record> records = getRecords(cypherQuery, params);
            records.forEach(record ->
                                    log.info("Updated object with element ID: {}", neo4jNode.getElementId()));
        } catch (Exception e) {
            log.error("Error updating object with element ID: {}", neo4jNode.getElementId(), e);
        }
    }

    /**
     * Deletes a node from the database using parameterized query.
     */
    public <T extends Neo4jNode> void deleteNode(T node) {
        try {
            String cypherQuery = """
                    MATCH (c)
                    WHERE elementId(c) = $elementId
                    DETACH DELETE c
                    """;

            Map<String, Object> params = Map.of("elementId", node.getElementId());
            Stream<Record> records = getRecords(cypherQuery, params);
            records.forEach(record ->
                                    log.info("Deleted object with element ID: {}", node.getElementId()));
        } catch (Exception e) {
            log.error("Error deleting object with element ID: {}", node.getElementId(), e);
        }
    }

    /**
     * Executes a custom Cypher query and maps the results to the specified node class.
     */
    public <T extends Neo4jNode> List<T> executeQuery(String query, Class<T> clazz) {
        return executeQuery(query, Map.of(), clazz);
    }

    /**
     * Executes a custom Cypher query with parameters and maps the results.
     */
    public <T extends Neo4jNode> List<T> executeQuery(
            String query,
            Map<String, Object> params,
            Class<T> clazz
    ) {
        if (query == null || query.isEmpty()) {
            log.error("Query is null or empty");
            return new ArrayList<>();
        }
        List<T> nodes = new ArrayList<>();
        try {
            Stream<Record> records = getRecords(query, params);
            records.forEach(record -> {
                NodeValue n = (NodeValue) record.get("n");
                T node = Neo4jNode.fromNodeValue(n, clazz);
                nodes.add(node);
            });
        } catch (Exception e) {
            log.error("Error executing query: {}", query, e);
        }
        return nodes;
    }

    public Stream<Record> executeQuery(
            String query
    ) {
        if (query == null || query.isEmpty()) {
            log.error("Query is null or empty");
            throw new IllegalArgumentException("Query cannot be null or empty");
        }
        return getRecords(query);
    }

    public <T extends Neo4jNode> List<Neo4jEmbeddingSearchResult<T>> similaritySearch(
            String embeddingIndex,
            List<Double> embeddings,
            int topK,
            Class<T> clazz
    ) {
        return similaritySearch(embeddingIndex, embeddings.stream().mapToDouble(Double::doubleValue).toArray(), topK,
                                clazz);
    }

    public <T extends Neo4jNode> List<Neo4jEmbeddingSearchResult<T>> similaritySearch(
            String embeddingIndex,
            List<Double> embeddings,
            int topK,
            Class<T> clazz,
            String parentId
    ) {
        return similaritySearch(embeddingIndex, embeddings.stream().mapToDouble(Double::doubleValue).toArray(), topK,
                                clazz, parentId);
    }

    public <T extends Neo4jNode> List<Neo4jEmbeddingSearchResult<T>> similaritySearch(
            String embeddingIndex,
            double[] embeddings,
            int topK,
            Class<T> clazz
    ) {
        return similaritySearch(embeddingIndex, embeddings, topK, clazz, null);
    }

    public <T extends Neo4jNode> List<Neo4jEmbeddingSearchResult<T>> similaritySearch(
            String embeddingIndex,
            double[] embeddings,
            int topK,
            Class<T> clazz,
            String parentId
    ) {
        String cypherQuery;
        Map<String, Object> params;
        if (parentId != null) {
            params = Map.of(
                    "embeddings", embeddings,
                    "k", topK,
                    "parentId", parentId
            );

            cypherQuery = """
                    MATCH (parent)-[]->(node:%s)
                    WHERE elementId(parent) = $parentId
                    WITH node,
                         gds.similarity.cosine(node.embedding, $embeddings) AS score
                    RETURN node, score
                    ORDER BY score DESC
                    LIMIT $k
                    """.formatted(
                    Neo4jNode.getLabel(clazz)
            );
        } else {
            params = Map.of(
                    "embeddings", embeddings,
                    "k", topK
            );

            cypherQuery = """
                    CALL db.index.vector.queryNodes('%s', $k, $embeddings)
                    YIELD node, score
                    RETURN node, score
                    ORDER BY score DESC
                    """.formatted(embeddingIndex);
        }

        List<Neo4jEmbeddingSearchResult<T>> results = new ArrayList<>();
        Stream<Record> records = getRecords(cypherQuery, params);
        records.forEach(record -> {
            NodeValue nodeValue = (NodeValue) record.get("node");
            double score = record.get("score").asDouble();
            Neo4jEmbeddingSearchResult<T> result = new Neo4jEmbeddingSearchResult<>();
            result.setNode(Neo4jNode.fromNodeValue(nodeValue, clazz));
            result.setScore(score);
            results.add(result);
        });
        return results;
    }

    /**
     * Executes a Cypher query and returns a stream of Neo4j records.
     */
    public Stream<Record> getRecords(String cypherQuery) {
        return getRecords(cypherQuery, Map.of());
    }

    /**
     * Executes a Cypher query with parameters and returns a stream of Neo4j records.
     */
    public Stream<Record> getRecords(String cypherQuery, Map<String, Object> params) {
        if (driver == null) {
            log.error("Neo4j driver is not initialized");
            throw new IllegalStateException("Neo4j driver is not initialized");
        }
        if (cypherQuery == null || cypherQuery.isEmpty()) {
            log.error("Cypher query is null or empty");
            throw new IllegalArgumentException("Cypher query cannot be null or empty");
        }
        return driver.session().run(cypherQuery, params).stream();
    }
}