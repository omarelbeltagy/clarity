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
import de.tum.claritypipeline.client.Client;
import de.tum.claritypipeline.model.relation.HasPatternProperties;
import de.tum.claritypipeline.model.relation.HasRagProperties;
import de.tum.claritypipeline.utils.EmbeddingUtils;
import de.tum.clarityutils.AfterDeserialization;
import de.tum.clarityutils.JacksonUtils;
import lombok.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Declarative LLM configuration referenced by every strategy (README “Model Properties”).
 * <p>Defines provider identity, prompting, sampling settings, response format and optional RAG/pattern helpers.
 * During initialization the appropriate {@link de.tum.claritypipeline.client.Client} is created and persisted.</p>
 */
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class ModelProperties extends Neo4jNode {

    /**
     * The model name
     */
    @JsonProperty("name")
    @JsonPropertyDescription("LLM model identifier as exposed by the provider (e.g., gpt-4.1, claude-sonnet-4.5).")
    private String name;

    /**
     * The model provider name (e.g., "openai", "anthropic").
     */
    @JsonProperty("provider")
    @JsonPropertyDescription("LLM provider key (openai, anthropic, together, local, ...).")
    private String provider;

    /**
     * The client instance used to interact with the language model.
     *
     * <p>Initialized after deserialization.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Client client;

    /**
     * The response format expected from the client.
     *
     * <p>Supported formats: JSON_OBJECT and TEXT (enumeration defined elsewhere).
     */
    @JsonProperty("response-format")
    @JsonPropertyDescription("The format of the classification response. Supported formats are JSON_OBJECT and TEXT.")
    private ResponseFormat responseFormat = ResponseFormat.JSON_OBJECT;

    /**
     * Maximum number of tokens allowed in the classification response.
     *
     * <p>Default is 4096.
     */
    @JsonProperty("max-tokens")
    @JsonPropertyDescription("The maximum number of tokens to generate in the classification response.")
    private int maxTokens = 4096;

    /**
     * Nucleus sampling parameter (top-p) for the language model.
     *
     * <p>Value is between 0 and 1.
     */
    @JsonProperty("top-p")
    @JsonPropertyDescription("The nucleus sampling parameter for the language model.")
    private Double topP;

    /**
     * Temperature parameter for the language model.
     *
     * <p>Higher values produce more creative outputs. Default is 0.9.
     */
    @JsonProperty("temperature")
    @JsonPropertyDescription("The temperature setting for the language model.")
    private Double temperature;

    /**
     * The prompt template or path to prompt file for classification.
     *
     * <p>If a file path is provided (ending with .yaml, .yml, or .txt), the prompt will be loaded from the file.
     * Otherwise, the prompt is used as-is.
     */
    @JsonProperty("prompt")
    @JsonPropertyDescription("The prompt template or path to prompt file for classification.")
    private String prompt;

    /**
     * Configuration for the pattern used to extract labels from textual responses.
     *
     * <p>Contains the regex and flags; defaults are provided by {@link PatternProperties}.
     */
    @JsonProperty("pattern")
    @JsonPropertyDescription("The pattern configuration for extracting labels from text. Contains the regex and flags.")
    @Neo4jIgnore
    @Setter(AccessLevel.NONE)
    @Getter(AccessLevel.NONE)
    private PatternProperties patternConfig;

    @JsonProperty("reasoning-effort")
    @JsonPropertyDescription("Set the reasoning effort for models that support that setting.")
    private String reasoningEffort;

    @JsonProperty("rag")
    @JsonPropertyDescription(
            "Optional Retrieval-Augmented Generation settings (examples injected per README RAG section).")
    @Neo4jIgnore
    private RagProperties ragProperties;

    /**
     * Compiled regex pattern for label extraction from text responses.
     *
     * <p>Initialized after deserialization.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Pattern pattern;

    private void createNode() {
        String query;
        Map<String, Object> propertiesMap = toPropertiesMap();
        if (reasoningEffort == null) {
            propertiesMap.put("reasoning-effort", null);
        }
        if (prompt != null) {
            propertiesMap.put("prompt", prompt);
        }
        String literal = toCypherMap(propertiesMap);
        if (ragProperties == null && patternConfig == null) {
            query = """
                    MATCH(n:%s %s)
                    WHERE NOT (n)-[:%s]->(:%s)
                        AND NOT (n)-[:%s]->(:%s)
                    """.formatted(
                    Neo4jNode.getLabel(ModelProperties.class),
                    literal,
                    Neo4jRelation.getType(HasRagProperties.class),
                    Neo4jNode.getLabel(RagProperties.class),
                    Neo4jRelation.getType(HasPatternProperties.class),
                    Neo4jNode.getLabel(PatternProperties.class)
            );
        } else if
        (ragProperties == null) {
            query = """
                    MATCH(n:%s %s)-[:%s]->(p:%s)
                    WHERE elementId(p) = '%s'
                        AND NOT (n)-[:%s]->(:%s)
                    """.formatted(
                    Neo4jNode.getLabel(ModelProperties.class),
                    literal,
                    Neo4jRelation.getType(HasPatternProperties.class),
                    Neo4jNode.getLabel(PatternProperties.class),
                    patternConfig.getElementId(),
                    Neo4jRelation.getType(HasRagProperties.class),
                    Neo4jNode.getLabel(RagProperties.class)
            );
        } else if (patternConfig == null) {
            query = """
                    MATCH(n:%s %s)-[:%s]->(r:%s)
                    WHERE elementId(r) = '%s'
                        AND NOT (n)-[:%s]->(:%s)
                    """.formatted(
                    Neo4jNode.getLabel(ModelProperties.class),
                    literal,
                    Neo4jRelation.getType(HasRagProperties.class),
                    Neo4jNode.getLabel(RagProperties.class),
                    ragProperties.getElementId(),
                    Neo4jRelation.getType(HasPatternProperties.class),
                    Neo4jNode.getLabel(PatternProperties.class)
            );
        } else {
            query = """
                    MATCH(r:%s)<-[:%s]-(n:%s %s)-[:%s]->(p:%s)
                    WHERE elementId(p) = '%s'
                        AND elementId(r) = '%s'
                    """.formatted(
                    Neo4jNode.getLabel(RagProperties.class),
                    Neo4jRelation.getType(HasRagProperties.class),
                    Neo4jNode.getLabel(ModelProperties.class),
                    literal,
                    Neo4jRelation.getType(HasPatternProperties.class),
                    Neo4jNode.getLabel(PatternProperties.class),
                    patternConfig.getElementId(),
                    ragProperties.getElementId()
            );
        }
        if (reasoningEffort == null) {
            query = """
                    %s
                        AND n.reasoningEffort IS NULL
                    """.formatted(query);
        }
        query = """
                %s
                RETURN n
                """.formatted(query);

        ModelProperties existingNode = GlobalConfig.NEO4J_CLIENT.executeQuery(query, ModelProperties.class).stream()
                                                                .findFirst()
                                                                .orElse(null);

        if (existingNode != null && allRelationsExist(existingNode)) {
            setElementId(existingNode.getElementId());
            return;
        }

        GlobalConfig.NEO4J_CLIENT.saveNode(this);
        createRelationIfNeeded(ragProperties, HasRagProperties.builder().build());
        createRelationIfNeeded(patternConfig, HasPatternProperties.builder().build());
    }

    /**
     * Enumeration of supported response formats returned by a classifier.
     *
     * <p>- JSON_OBJECT: structured JSON response that can be parsed into fields.<br>
     * - TEXT: plain text response that may require regex extraction.
     */
    public enum ResponseFormat {
        /**
         * Indicates the classifier returns a structured JSON object.
         *
         * <p>Serialized as "json_object".
         */
        @JsonProperty("json_object")
        JSON_OBJECT,

        /**
         * Indicates the classifier returns plain textual output.
         *
         * <p>Serialized as "text".
         */
        @JsonProperty("text")
        TEXT
    }

    /**
     * Configuration holder for a regex pattern used to extract labels from client text responses.
     *
     * <p>Contains the regex string and a pipe-separated list of human-friendly flag names which are mapped
     * to {@link Pattern} constants by {@link #getFlagsMask()}.
     */
    @Node(label = "PatternProperties")
    @Getter
    @Setter
    public static class PatternProperties extends Neo4jNode {

        /**
         * Mapping from human-friendly flag names to java.util.regex.Pattern flags.
         *
         * <p>Keys are normalized (lowercase, hyphenated) versions of user-provided flag names.
         */
        private static final Map<String, Integer> FLAG_MAPPINGS = Map.of(
                "case-insensitive", Pattern.CASE_INSENSITIVE,
                "multiline", Pattern.MULTILINE,
                "dotall", Pattern.DOTALL,
                "unicode-case", Pattern.UNICODE_CASE,
                "canon-eq", Pattern.CANON_EQ,
                "unix-lines", Pattern.UNIX_LINES,
                "literal", Pattern.LITERAL,
                "unicode-character-class", Pattern.UNICODE_CHARACTER_CLASS,
                "comments", Pattern.COMMENTS
        );

        /**
         * The regular expression used to extract the label from a textual response.
         *
         * <p>Default: "^Label:\s*(.+)$"
         */
        @JsonProperty("regex")
        @JsonPropertyDescription("The regex pattern to extract labels from text responses.")
        private String regex = "^Label:\\s*(.+)$";

        /**
         * Pipe-separated list of flag names to enable for the regex, e.g. "multiline|case-insensitive".
         *
         * <p>Default: "multiline"
         */
        @JsonProperty("flags")
        @JsonPropertyDescription("The regex flags to use, separated by '|'. E.g., 'multiline|case-insensitive'.")
        private String flags = "multiline";

        public static PatternProperties load(String path) throws IOException {
            if (path == null || path.isEmpty()) {
                throw new IOException("No path specified for PatternProperties file.");
            }
            ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
            return JacksonUtils.readAndInit(mapper, new File(path),
                                            PatternProperties.class);
        }

        /**
         * Convert the human-friendly flags string into an integer mask appropriate for {@link Pattern}.
         *
         * <p>If flags is null or empty, {@link Pattern#MULTILINE} is returned by default.
         *
         * @return combined int mask of Pattern flags
         */
        public int getFlagsMask() {
            if (flags == null || flags.isEmpty()) {
                return Pattern.MULTILINE;
            }

            return Arrays.stream(flags.split("\\|"))
                         .map(String::trim)
                         .map(f -> f.toLowerCase().replace("_", "-"))
                         .map(FLAG_MAPPINGS::get)
                         .filter(Objects::nonNull)
                         .reduce(0, (a, b) -> a | b);
        }

        @AfterDeserialization
        public void initialize() {
            PatternProperties patternProperties = GlobalConfig.NEO4J_CLIENT.findNode(toPropertiesMap(),
                                                                                     PatternProperties.class);
            if (patternProperties != null) {
                this.setElementId(patternProperties.getElementId());
                return;
            }
            GlobalConfig.NEO4J_CLIENT.saveNode(this);
        }
    }

    @JsonSetter("pattern")
    public void setPattern(Object raw) throws IOException {
        if (raw instanceof String s) {
            this.patternConfig = PatternProperties.load(s);
            return;
        }
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        this.patternConfig = JacksonUtils.convertAndInit(mapper, raw, PatternProperties.class);

        int flags = patternConfig.getFlagsMask();
        this.pattern = Pattern.compile(patternConfig.getRegex(), flags);
    }

    /**
     * Sets the prompt, loading from file if a file path is provided.
     *
     * <p>If the prompt ends with .yaml or .yml, it is loaded as a YAML file containing a "prompt" key.
     * If it ends with .txt, it is loaded as a plain text file.
     * Otherwise, the prompt is used as-is.
     *
     * @param prompt The prompt string or file path.
     */
    @JsonSetter("prompt")
    public void setPrompt(String prompt) {
        if (prompt == null || prompt.isEmpty()) {
            return;
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
                this.prompt = promptFile.get("prompt");
                return;
            } catch (IOException e) {
                throw new RuntimeException("Failed to load prompt from file: " + prompt, e);
            }
        }
        if (prompt.endsWith(".txt")) {
            try {
                this.prompt = Files.readString(java.nio.file.Path.of(prompt));
                return;
            } catch (IOException e) {
                throw new RuntimeException("Failed to load prompt from file: " + prompt, e);
            }
        }
        this.prompt = prompt;
    }

    /**
     * Initializes the model configuration after deserialization.
     *
     * <p>Validates required fields, compiles the regex pattern, and creates the client instance.
     */
    @AfterDeserialization
    public void initialize() {
        if (provider == null || provider.isEmpty()) {
            throw new IllegalArgumentException("Model provider must be specified in Model configuration.");
        }
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("Model name must be specified in Model configuration.");
        }
        if (patternConfig == null && responseFormat == ResponseFormat.TEXT) {
            throw new IllegalArgumentException("Pattern configuration must be specified.");
        }
        this.client = Client.create(this);
        createNode();
    }

    @Node(label = "RagProperties")
    @Getter
    @Setter
    public static class RagProperties extends Neo4jNode {
        @JsonProperty("enabled")
        @JsonPropertyDescription("Whether dynamic few-shot examples should be retrieved from the embedding index.")
        private boolean enabled;

        @JsonProperty("embedding-index")
        @JsonPropertyDescription("Name of the Neo4j embedding index that stores QA vectors for retrieval.")
        private EmbeddingIndex embeddingIndex;

        @JsonProperty("k")
        @JsonPropertyDescription("Number of nearest-neighbour examples to fetch per taxonomy category.")
        private int k = 1;

        @AfterDeserialization
        public void initialize() {
            if (enabled) {
                if (embeddingIndex == null) {
                    throw new IllegalArgumentException("RAG is enabled but embedding index is not set.");
                }
                RagProperties raqProperties = GlobalConfig.NEO4J_CLIENT.findNode(toPropertiesMap(),
                                                                                 RagProperties.class);
                EmbeddingUtils.ensureEmbeddingIndicesExist(GlobalConfig.NEO4J_CLIENT);
                if (raqProperties != null) {
                    this.setElementId(raqProperties.getElementId());
                    return;
                }
                GlobalConfig.NEO4J_CLIENT.saveNode(this);
            }
        }
    }

    private boolean allRelationsExist(ModelProperties existingNode) {
        boolean ragRelationOk = ragProperties == null ||
                GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(), ragProperties.getElementId(),
                                                       HasRagProperties.class)
                        != null;

        boolean patternRelationOk = patternConfig == null ||
                GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(), patternConfig.getElementId(),
                                                       HasPatternProperties.class) != null;

        return ragRelationOk && patternRelationOk;
    }

    private <T extends Neo4jRelation, N extends Neo4jNode> void createRelationIfNeeded(
            N targetNode, T relation) {
        if (targetNode == null) return;
        relation.setStartNodeId(this.getElementId());
        relation.setEndNodeId(targetNode.getElementId());
        GlobalConfig.NEO4J_CLIENT.createRelation(relation);
    }
}
