package de.tum.claritypipeline.model.properties;

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
import de.tum.claritypipeline.model.Classification;
import de.tum.claritypipeline.model.Cluster;
import de.tum.claritypipeline.model.Evaluation;
import de.tum.claritypipeline.model.Taxonomy;
import de.tum.claritypipeline.model.relation.HasEvaluation;
import de.tum.claritypipeline.service.EmbeddingService;
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
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
public class ClassificationProperties extends Neo4jNode implements Serializable {

    /**
     * The name of the classification task.
     *
     * <p>This is a required field and used to identify the classification run.
     */
    @JsonProperty("name")
    @JsonPropertyDescription("The name of the classification task.")
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
     * File path to the Neo4j credentials YAML file.
     *
     * <p>Not persisted on the Neo4j node; used to load {@link #neo4jCredentials} at runtime.
     */
    @JsonProperty("neo4j-credentials")
    @JsonPropertyDescription("The file path to the Neo4j credentials YAML file.")
    @Getter(AccessLevel.NONE)
    @Neo4jIgnore
    private String neo4jCredentialsFile;

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
    @JsonProperty("strategy")
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
     * The name of the cluster to which this classification run belongs.
     *
     * <p>Used for organizing multiple classification runs under a common cluster.
     */
    @JsonProperty("cluster")
    @JsonPropertyDescription("The cluster name for organizing classification runs.")
    @Neo4jIgnore
    @Getter(AccessLevel.NONE)
    private String clusterName;

    /**
     * The embedding model to use for generating embeddings.
     *
     * <p>Default is "text-embedding-3-small".
     */
    @JsonProperty("embedding-model")
    @JsonPropertyDescription("The embedding model to use for generating embeddings if it is used for the strategy.")
    private String embeddingModel = "text-embedding-3-small";

    /**
     * File path to the taxonomy YAML file used for mapping or validating labels.
     *
     * <p>Loaded at initialization into {@link #taxonomy}.
     */
    @JsonProperty("taxonomy")
    @JsonPropertyDescription("The file path to the taxonomy YAML file used for classification.")
    @Setter(AccessLevel.NONE)
    @Getter(AccessLevel.NONE)
    @Neo4jIgnore
    private String taxonomyFile;

    /**
     * Timestamp when this properties instance was created/loaded.
     *
     * <p>Stored as ISO-8601 string; initialized to current UTC time by default.
     */
    @JsonIgnore
    private String timestamp = String.valueOf(OffsetDateTime.now(ZoneOffset.UTC));

    /**
     * Neo4j credentials object loaded from {@link #neo4jCredentialsFile} or defaults.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Neo4jCredentials neo4jCredentials;

    /**
     * A {@link Classification} node instance used to group or identify this run in Neo4j.
     *
     * <p>Constructed during initialization.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Classification classification;

    /**
     * The cluster information of the classification to organize multiple Classification runs.
     *
     * <p>Loaded from {@link #neo4jCredentials} at initialization.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Cluster cluster;

    /**
     * The taxonomy loaded from {@link #taxonomyFile} used for mapping labels.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Taxonomy taxonomy;

    /**
     * The associated {@link Evaluation} node linked via a HasEvaluation relationship.
     *
     * <p>Loaded on-demand from Neo4j.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Evaluation evaluation;

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

    /**
     * Set the taxonomy file path and load the taxonomy.
     *
     * @param taxonomyFile path to the taxonomy YAML file
     */
    @JsonSetter("taxonomy")
    public void setTaxonomyFile(String taxonomyFile) {
        this.taxonomyFile = taxonomyFile;
        if (taxonomyFile == null || taxonomyFile.isEmpty()) {
            throw new IllegalArgumentException("Taxonomy file path must be provided.");
        }
        try {
            this.taxonomy = Taxonomy.load(taxonomyFile);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load taxonomy file: " + taxonomyFile, e);
        }
    }

    /**
     * Initialize derived fields after deserialization.
     *
     * <p>This method validates required fields and constructs
     * {@link #classification} and {@link #cluster} nodes.
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
        if (query == null || query.isEmpty()) {
            throw new IOException("Missing query for classification properties");
        }
        if (strategy == null) {
            throw new IOException("Missing strategy for classification properties");
        }
        if (taxonomy == null) {
            throw new IOException("Missing taxonomy for classification properties");
        }

        if (neo4jCredentialsFile == null || neo4jCredentialsFile.isEmpty()) {
            neo4jCredentials = Neo4jCredentials.getDefault();
        } else {
            neo4jCredentials = Neo4jCredentials.load(neo4jCredentialsFile);
        }

        EmbeddingService.initialize(neo4jCredentials, embeddingModel);

        this.classification = Classification.builder()
                                            .name(name)
                                            .build();

        if (clusterName != null && !clusterName.isEmpty()) {
            this.cluster = Cluster.builder()
                                  .name(clusterName)
                                  .build();
        }
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

