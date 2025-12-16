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
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.claritypipeline.model.core.Taxonomy;
import de.tum.claritypipeline.model.relation.HasClassificationModel;
import de.tum.claritypipeline.model.relation.HasTaxonomy;
import de.tum.clarityutils.AfterDeserialization;
import de.tum.clarityutils.JacksonUtils;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.Map;

@Node(label = "PromptEnhancingProperties")
@Getter
@Setter
public class PromptEnhancingProperties extends Neo4jNode implements Serializable {

    @JsonProperty(value = "name", index = 4)
    private String name;

    @JsonProperty("version")
    private String version;

    @JsonProperty("output-prompt")
    private String outputPrompt;

    @JsonProperty("output-taxonomy")
    private String outputTaxonomy;

    @JsonProperty("query")
    @JsonPropertyDescription("The Cypher query to specify which items to fetch for the prompt enhancing.")
    private String query;

    @JsonProperty(value = "neo4j-credentials", index = 0)
    @JsonPropertyDescription("The neo4j credentials configuration.")
    @Neo4jIgnore
    @Setter(AccessLevel.NONE)
    private Neo4jCredentials neo4jCredentials = Neo4jCredentials.getDefault();

    /**
     * The classification strategy to use for this run.
     *
     * <p>Defines how the classification is performed (e.g., zero-shot, few-shot).
     */
    @JsonProperty(value = "model", index = 1)
    @JsonPropertyDescription("The classification strategy to use for this classification run.")
    @Neo4jIgnore
    private ModelProperties model;

    /**
     * Number of iterations to enhance the prompt.
     */
    @JsonProperty("iterations")
    @JsonPropertyDescription("The number of iterations to enhance the prompt.")
    private Integer iterations = 5;

    @JsonProperty("worker-threads")
    @JsonPropertyDescription("The number of worker threads for parallel classification.")
    @Neo4jIgnore
    private int workerThreads = 12;

    @JsonProperty("classification-prompt")
    @Setter(AccessLevel.NONE)
    private String classificationPrompt;

    @JsonProperty("enhancement-prompt-diagnose")
    @Setter(AccessLevel.NONE)
    private String enhancementPromptDiagnose;

    @JsonProperty("enhancement-prompt-patch")
    @Setter(AccessLevel.NONE)
    private String enhancementPromptPatch;

    /**
     * Number of QA pairs to fetch for prompt enhancement.
     */
    @JsonProperty("n")
    @JsonPropertyDescription("The number of QA pairs to fetch for prompt enhancement.")
    private Integer n = 20;

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

    public PromptEnhancingProperties() throws IOException {}

    /**
     * Read YAML and map it into a {@link PromptEnhancingProperties} instance.
     *
     * <p>Uses Jackson YAML mapper with case-insensitive enum deserialization.
     *
     * @param path path to the YAML file
     * @return deserialized ClassificationProperties
     * @throws IOException if reading or mapping fails
     */
    public static PromptEnhancingProperties load(String path) throws IOException {
        if (path == null || path.isEmpty()) {
            throw new IOException("No path specified for ClassificationProperties file.");
        }
        ObjectMapper mapper = JsonMapper.builder(new YAMLFactory())
                                        .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_ENUMS, true)
                                        .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, false)
                                        .build();
        return JacksonUtils.readAndInit(mapper, new File(path), PromptEnhancingProperties.class);
    }

    private void createNode() {
        Map<String, Object> props = Map.of("name", name, "version", version);
        PromptEnhancingProperties existingNode = GlobalConfig.NEO4J_CLIENT.findNode(props,
                                                                                    PromptEnhancingProperties.class);

        if (existingNode == null) {
            GlobalConfig.NEO4J_CLIENT.saveNode(this);
            createRelationIfNeeded(taxonomy, HasTaxonomy.builder().build());
            createRelationIfNeeded(model, HasClassificationModel.builder().build());
            return;
        }
        throw new RuntimeException(
                "PromptEnhancingProperties node with name '" + name + "' and version '" + version +
                        "' already exists in the database.");
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
     * Initialize derived fields after deserialization.
     *
     * <p>This method validates required fields and constructs
     *
     * @throws IOException if required fields are missing
     */
    @AfterDeserialization
    private void initialize() throws IOException {
        if (name == null || name.isEmpty()) {
            throw new IOException("Missing name for prompt enhancing properties");
        }
        if (version == null || version.isEmpty()) {
            throw new IOException("Missing version for prompt enhancing properties");
        }
        if (neo4jCredentials == null) {
            throw new IOException("Missing neo4-credentials for prompt enhancing properties");
        }
        if (query == null || query.isEmpty()) {
            throw new IOException("Missing query for prompt enhancing properties");
        }
        if (taxonomy == null) {
            throw new IOException("Missing taxonomy for prompt enhancing properties");
        }
        if (model == null) {
            throw new IOException("Missing model for prompt enhancing properties");
        }
        createNode();
    }

    @JsonSetter("classification-prompt")
    public void setClassificationPrompt(String prompt) {
        this.classificationPrompt = readPrompt(prompt);
    }

    @JsonSetter("enhancement-prompt-diagnose")
    public void setEnhancementPrompt(String prompt) {
        this.enhancementPromptDiagnose = readPrompt(prompt);
    }

    @JsonSetter("enhancement-prompt-patch")
    public void setEnhancementPromptPatch(String prompt) {
        this.enhancementPromptPatch = readPrompt(prompt);
    }

    private String readPrompt(String prompt) {
        if (prompt == null || prompt.isEmpty()) {
            return null;
        }
        if (prompt.endsWith(".yaml") || prompt.endsWith(".yml")) {
            try {
                ObjectMapper mapper = JsonMapper.builder(new YAMLFactory())
                                                .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_ENUMS, true)
                                                .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, false)
                                                .build();
                Map<String, String> promptFile = mapper.readValue(new File(prompt), Map.class);
                if (!promptFile.containsKey("prompt")) {
                    throw new IOException("Prompt file does not contain 'prompt' key: " + prompt);
                }
                return promptFile.get("prompt");
            } catch (IOException e) {
                throw new RuntimeException("Failed to load prompt from file: " + prompt, e);
            }
        }
        if (prompt.endsWith(".txt")) {
            try {
                return Files.readString(java.nio.file.Path.of(prompt));
            } catch (IOException e) {
                throw new RuntimeException("Failed to load prompt from file: " + prompt, e);
            }
        }
        return prompt;
    }

    private <T extends Neo4jRelation, N extends Neo4jNode> void createRelationIfNeeded(
            N targetNode, T relation) {
        if (targetNode == null || targetNode.getElementId() == null) return;
        relation.setStartNodeId(this.getElementId());
        relation.setEndNodeId(targetNode.getElementId());
        GlobalConfig.NEO4J_CLIENT.createRelation(relation);
    }
}

