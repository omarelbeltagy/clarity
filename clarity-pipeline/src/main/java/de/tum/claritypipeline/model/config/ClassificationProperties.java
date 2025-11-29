package de.tum.claritypipeline.model.config;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.claritypipeline.model.core.Taxonomy;
import de.tum.claritypipeline.model.relation.HasClassificationStrategy;
import de.tum.claritypipeline.model.relation.HasEvaluation;
import de.tum.claritypipeline.model.relation.HasTaxonomy;
import de.tum.claritypipeline.model.relation.IsPartOf;
import de.tum.claritypipeline.strategy.ClassificationStrategy;
import de.tum.clarityutils.AfterDeserialization;
import de.tum.clarityutils.JacksonUtils;
import lombok.*;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;

/**
 * Represents the configuration and runtime properties for a classification task.
 *
 * <p>This class is populated from a YAML file and contains both persisted configuration
 * fields (serialized) and transient runtime objects (ignored for serialization).
 *
 * <p>Usage:
 * - Load an instance from a YAML file using {@link #load(String)} which validates and
 * initializes necessary fields.
 *
 * <p>The class extends {@link Neo4jNode} and is annotated with {@link Node} for Neo4j mapping.
 */
@Node(label = "ClassificationProperties")
@Getter
@Setter
public class ClassificationProperties extends Neo4jNode implements Serializable {

    /**
     * The name of the classification task.
     *
     * <p>This is a required field and used to identify the classification run.
     */
    @JsonProperty(value = "name", index = 4)
    @JsonPropertyDescription("The name of the classification task.")
    @Setter(AccessLevel.NONE)
    private String name;

    /**
     * The version of the current classification run.
     *
     * <p>Used to track configuration or schema versions for reproducibility.
     */
    @JsonProperty("version")
    @JsonPropertyDescription("The version of the current classification run.")
    private String version;

    /**
     * The Cypher query used to fetch items for classification from Neo4j.
     *
     * <p>Should return the nodes/records which will be classified by the configured client.
     */
    @JsonProperty("query")
    @JsonPropertyDescription("The Cypher query to specify which items to fetch for classification.")
    private String query;

    /**
     * The Neo4j Credentials
     *
     * <p>Not persisted on the Neo4j node.
     */
    @JsonProperty(value = "neo4j-credentials", index = 0)
    @JsonPropertyDescription("The neo4j credentials configuration.")
    @Neo4jIgnore
    @Setter(AccessLevel.NONE)
    private Neo4jCredentials neo4jCredentials = Neo4jCredentials.getDefault();

    /**
     * Number of worker threads to use for parallel classification.
     *
     * <p>Default is 12.
     */
    @JsonProperty("worker-threads")
    @JsonPropertyDescription("The number of worker threads for parallel classification.")
    @Neo4jIgnore
    private int workerThreads = 12;

    /**
     * The classification strategy to use for this run.
     *
     * <p>Defines how the classification is performed (e.g., zero-shot, few-shot).
     */
    @JsonProperty(value = "strategy", index = 1)
    @JsonPropertyDescription("The classification strategy to use for this classification run.")
    @Neo4jIgnore
    private ClassificationStrategy strategy;

    /**
     * Number of attempts to classify an item in case of transient failures.
     *
     * <p>Default is 5.
     */
    @JsonProperty("attempts")
    @JsonPropertyDescription("The number of attempts to classify an item in case of failure.")
    private int attempts = 5;

    /**
     * File path to the taxonomy YAML file used for mapping or validating labels.
     */
    @JsonProperty(value = "taxonomy", index = 2)
    @JsonPropertyDescription("The taxonomy used for classification.")
    @Setter(AccessLevel.NONE)
    @Neo4jIgnore
    private Taxonomy taxonomy;

    @JsonIgnore
    private String firstStartedAt = String.valueOf(OffsetDateTime.now(ZoneOffset.UTC));

    @JsonIgnore
    private String lastStartedAt = String.valueOf(OffsetDateTime.now(ZoneOffset.UTC));

    /**
     * A {@link Classification} node instance used to group or identify this run in Neo4j.
     *
     * <p>Constructed during initialization.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Classification classification;

    /**
     * The associated {@link Evaluation} node linked via a HasEvaluation relationship.
     *
     * <p>Loaded on-demand from Neo4j.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Evaluation evaluation;

    public ClassificationProperties() throws IOException {}

    private void createNode() {
        Map<String, Object> props = Map.of("name", name, "version", version);
        ClassificationProperties existingNode = GlobalConfig.NEO4J_CLIENT.findNode(props,
                                                                                   ClassificationProperties.class);

        if (existingNode == null) {
            GlobalConfig.NEO4J_CLIENT.saveNode(this);
            createRelationIfNeeded(classification, IsPartOf.builder().build());
            createRelationIfNeeded(taxonomy, HasTaxonomy.builder().build());
            createRelationIfNeeded(strategy.getClassificationStrategyNode(),
                                   HasClassificationStrategy.builder().build());
            return;
        }
        if (allRelationsExist(existingNode)) {
            this.setElementId(existingNode.getElementId());
            this.setFirstStartedAt(existingNode.getFirstStartedAt());
            GlobalConfig.NEO4J_CLIENT.updateNode(this);
        } else {
            throw new RuntimeException(
                    "The classification %s with version %s already exists but with different properties.".formatted(
                            name, version));
        }
    }

    /**
     * Read YAML and map it into a {@link ClassificationProperties} instance.
     *
     * <p>Uses Jackson YAML mapper with case-insensitive enum deserialization.
     *
     * @param path path to the YAML file
     * @return deserialized ClassificationProperties
     * @throws IOException if reading or mapping fails
     */
    public static ClassificationProperties load(String path) throws IOException {
        if (path == null || path.isEmpty()) {
            throw new IOException("No path specified for ClassificationProperties file.");
        }
        ObjectMapper mapper = JsonMapper.builder(new YAMLFactory())
                                        .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_ENUMS, true)
                                        .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, false)
                                        .build();
        return JacksonUtils.readAndInit(mapper, new File(path), ClassificationProperties.class);
    }

    @JsonSetter("name")
    public void setName(Object raw) throws IOException {
        if (raw instanceof String s) {
            this.name = s;
            this.classification = Classification.builder()
                                                .name(name)
                                                .build();
            Classification existingNode = GlobalConfig.NEO4J_CLIENT.findNode(Map.of("name", this.name),
                                                                             Classification.class);
            if (existingNode != null) {
                classification.setElementId(existingNode.getElementId());
                return;
            }
            GlobalConfig.NEO4J_CLIENT.saveNode(classification);
            return;
        }
        throw new IOException("Classification name must be a String");
    }

    @JsonSetter("taxonomy")
    public void setTaxonomy(Object raw) throws IOException {
        if (raw instanceof String s) {
            this.taxonomy = Taxonomy.load(s);
            return;
        }
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        this.taxonomy = mapper.convertValue(raw, Taxonomy.class);
    }

    @JsonSetter("neo4j-credentials")
    public void setNeo4jCredentials(Object raw) throws IOException {
        if (raw instanceof String s) {
            this.neo4jCredentials = Neo4jCredentials.load(s);
            return;
        }
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        this.neo4jCredentials = mapper.convertValue(raw, Neo4jCredentials.class);
        GlobalConfig.NEO4J_CREDENTIALS = this.neo4jCredentials;
    }

    /**
     * Holds evaluation metrics for a classification experiment.
     *
     * <p>Metrics are typical classification measures such as accuracy, precision, recall and F1 scores.
     */
    @Node(label = "Evaluation")
    @Getter
    @Setter
    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Evaluation extends Neo4jNode {

        /**
         * Overall accuracy of the classifier (correct / total).
         */
        private double accuracy;

        /**
         * Precision metric (positive predictive value).
         */
        private double precision;

        /**
         * Recall metric (sensitivity).
         */
        private double recall;

        /**
         * Macro-averaged F1 score across classes.
         */
        private double macroF1;

        /**
         * Macro-averaged F1 score across classes, rounded to 2 decimal places.
         */
        private double macroF1Rounded;

        /**
         * Micro-averaged F1 score across classes.
         */
        private double microF1;
    }

    /**
     * Initialize derived fields after deserialization.
     *
     * <p>This method validates required fields and constructs
     *
     * @throws IOException if required fields are missing
     */
    @AfterDeserialization
    private void initialize() throws IOException {
        if (name == null || name.isEmpty()) {
            throw new IOException("Missing name for classification properties");
        }
        if (version == null || version.isEmpty()) {
            throw new IOException("Missing version for classification properties");
        }
        if (neo4jCredentials == null) {
            throw new IOException("Missing neo4-credentials for classification properties");
        }
        if (query == null || query.isEmpty()) {
            throw new IOException("Missing query for classification properties");
        }
        if (strategy == null) {
            throw new IOException("Missing strategy for classification properties");
        }
        if (taxonomy == null) {
            throw new IOException("Missing taxonomy for classification properties");
        }
        createNode();
    }

    @Node(label = "Classification")
    @Getter
    @Setter
    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Classification extends Neo4jNode {

        /**
         * The name of the classification.
         *
         * <p>Used as an identifier when creating run nodes in Neo4j.
         */
        private String name;

        /**
         * The classification runs connected to this classification.
         */
        @JsonIgnore
        @Neo4jIgnore
        private List<ClassificationProperties> runs;

        /**
         * Retrieves the classification properties connected to this classification via "IS_PART_OF" relationships.
         *
         * @param neo4jClient The Neo4j client used to execute the query.
         * @return A list of ClassificationProperties nodes connected to this classification.
         */
        public List<ClassificationProperties> getRuns(Neo4jClient neo4jClient) {
            if (this.runs != null) {
                return this.runs;
            } else {
                if (getElementId() == null) {
                    return List.of();
                } else {
                    String query = String.format("""
                                                         MATCH (n:%s)-[:%s]->(u:%s)
                                                         WHERE elementId(u) = '%s'
                                                         RETURN n
                                                         """,
                                                 Neo4jNode.getLabel(ClassificationProperties.class),
                                                 Neo4jRelation.getType(IsPartOf.class),
                                                 Neo4jNode.getLabel(Classification.class),
                                                 getElementId()
                    );
                    List<ClassificationProperties> children = neo4jClient.executeQuery(query,
                                                                                       ClassificationProperties.class);
                    this.runs = children;
                    return children;
                }
            }
        }
    }

    private boolean allRelationsExist(ClassificationProperties existingNode) {
        boolean classificationRelationOk =
                GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(), classification.getElementId(),
                                                       IsPartOf.class)
                        != null;

        boolean taxonomyRelationOk =
                GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(), taxonomy.getElementId(),
                                                       HasTaxonomy.class) != null;

        boolean strategyRelationOk = strategy.getClassificationStrategyNode().getElementId() == null ||
                GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(),
                                                       strategy.getClassificationStrategyNode().getElementId(),
                                                       HasClassificationStrategy.class) != null;

        return classificationRelationOk && taxonomyRelationOk && strategyRelationOk;
    }

    private <T extends Neo4jRelation, N extends Neo4jNode> void createRelationIfNeeded(
            N targetNode, T relation) {
        if (targetNode == null || targetNode.getElementId() == null) return;
        relation.setStartNodeId(this.getElementId());
        relation.setEndNodeId(targetNode.getElementId());
        GlobalConfig.NEO4J_CLIENT.createRelation(relation);
    }


    /**
     * Retrieve the associated {@link Evaluation} node from Neo4j if not already loaded.
     *
     * <p>This method queries Neo4j for the Evaluation node linked via a HasEvaluation relationship.
     * If found, it caches the result in {@link #evaluation}.
     *
     * @param neo4jClient the Neo4j client to execute the query
     * @return the associated Evaluation node, or null if none exists
     */
    public Evaluation getEvaluation(Neo4jClient neo4jClient) {
        if (this.evaluation != null) {
            return this.evaluation;
        } else {
            if (getElementId() == null) {
                return null;
            } else {
                String query = String.format("""
                                                     MATCH (n:%s)<-[:%s]-(u:%s)
                                                     WHERE elementId(u) = '%s'
                                                     RETURN n
                                                     """,
                                             Neo4jNode.getLabel(Evaluation.class),
                                             Neo4jRelation.getType(HasEvaluation.class),
                                             Neo4jNode.getLabel(ClassificationProperties.class),
                                             getElementId()
                );
                List<Evaluation> children = neo4jClient.executeQuery(query,
                                                                     Evaluation.class);

                if (children.isEmpty()) {
                    return null;
                } else {
                    this.evaluation = children.getFirst();
                    return this.evaluation;
                }
            }
        }
    }
}

