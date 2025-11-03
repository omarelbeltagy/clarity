package de.tum.claritypipeline.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
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
import de.tum.claritypipeline.client.*;
import de.tum.claritypipeline.model.relation.HasEvaluation;
import lombok.*;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Represents the configuration and runtime properties for a classification task.
 *
 * <p>This class is populated from a YAML file and contains both persisted configuration
 * fields (serialized) and transient runtime objects (ignored for serialization).
 *
 * <p>Usage:
 * - Load an instance from a YAML file using {@link #load(String)} which validates and
 * initializes derived fields (e.g., {@link #client}, {@link #pattern}, {@link #taxonomy}).
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
     * The prompt template used when querying the language model for classification.
     *
     * <p>This should be a template that the chosen client can consume.
     */
    @JsonProperty("prompt")
    @JsonPropertyDescription("The prompt template used for classification.")
    private String prompt;

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
     * Model selector string in the format "provider:model_name".
     *
     * <p>It is parsed to determine the {@link ModelProvider} and the actual model name
     * used by the client. Saved in YAML as "model".
     */
    @JsonProperty("model")
    @JsonPropertyDescription("The model to use for classification, specified as 'provider:model_name'.")
    @Getter(AccessLevel.NONE)
    @Neo4jIgnore
    private String modelField;

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
     * The Cypher query used to fetch items for classification from Neo4j.
     *
     * <p>Should return the nodes/records which will be classified by the configured client.
     */
    @JsonProperty("query")
    @JsonPropertyDescription("The Cypher query to specify which items to fetch for classification.")
    private String query;

    /**
     * Number of attempts to classify an item in case of transient failures.
     *
     * <p>Default is 5.
     */
    @JsonProperty("attempts")
    @JsonPropertyDescription("The number of attempts to classify an item in case of failure.")
    private int attempts = 5;

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
     * <p>Value is between 0 and 1. Default is 0.5.
     */
    @JsonProperty("top-p")
    @JsonPropertyDescription("The nucleus sampling parameter for the language model.")
    private double topP = 0.5;

    /**
     * Temperature parameter for the language model.
     *
     * <p>Higher values produce more creative outputs. Default is 0.7.
     */
    @JsonProperty("temperature")
    @JsonPropertyDescription("The temperature setting for the language model.")
    private double temperature = 0.7;

    /**
     * The response format expected from the client.
     *
     * <p>Supported formats: JSON_OBJECT and TEXT (enumeration defined elsewhere).
     */
    @JsonProperty("response-format")
    @JsonPropertyDescription("The format of the classification response. Supported formats are JSON_OBJECT and TEXT.")
    @Neo4jIgnore
    private ResponseFormat responseFormat = ResponseFormat.JSON_OBJECT;

    @JsonProperty("cluster")
    @JsonPropertyDescription("The cluster name for organizing classification runs.")
    @Neo4jIgnore
    @Getter(AccessLevel.NONE)
    private String clusterName;

    /**
     * Whether the client is expected to return structured output.
     *
     * <p>Only meaningful when {@link #responseFormat} is JSON_OBJECT and supported by the provider.
     */
    @JsonProperty("structured-output")
    @JsonPropertyDescription(
            "Whether to expect structured output from the classifier. Only available for JSON_OBJECT response format "
                    + "and limited model providers.")
    @Neo4jIgnore
    private boolean structuredOutput = true;

    /**
     * File path to the taxonomy YAML file used for mapping or validating labels.
     *
     * <p>Loaded at initialization into {@link #taxonomy}.
     */
    @JsonProperty("taxonomy")
    @JsonPropertyDescription("The file path to the taxonomy YAML file used for classification.")
    @Getter(AccessLevel.NONE)
    @Neo4jIgnore
    private String taxonomyFile;

    /**
     * Configuration for the pattern used to extract labels from textual responses.
     *
     * <p>Contains the regex and flags; defaults are provided by {@link PatternConfig}.
     */
    @JsonProperty("pattern")
    @JsonPropertyDescription("The pattern configuration for extracting labels from text. Contains the regex and flags.")
    @Neo4jIgnore
    private PatternConfig patternConfig;

    /**
     * Whether to automatically inject response format instructions into the prompt.
     *
     * <p>If true, the system will modify the prompt to include instructions on the expected response format.
     * Default is true.
     */
    @JsonProperty("inject-response-format-in-prompt")
    @JsonPropertyDescription(
            "Whether to inject the response format instructions automatically in the prompt sent to the model.")
    @Neo4jIgnore
    private boolean injectResponseFormatInPrompt = true;

    /**
     * The parsed model name (without provider prefix).
     *
     * <p>Derived at initialization from {@link #modelField}.
     */
    @JsonIgnore
    private String model;

    /**
     * The model provider name (e.g., "openai", "anthropic").
     *
     * <p>Derived at initialization from {@link #modelField}.
     */
    @JsonIgnore
    private String provider;

    /**
     * Timestamp when this properties instance was created/loaded.
     *
     * <p>Stored as ISO-8601 string; initialized to current UTC time by default.
     */
    @JsonIgnore
    private String timestamp = String.valueOf(OffsetDateTime.now(ZoneOffset.UTC));

    /**
     * Compiled regex {@link Pattern} instantiated from {@link #patternConfig}.
     *
     * <p>Initialized during {@link #load(String)}.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Pattern pattern;

    /**
     * The runtime {@link Client} instance created for the configured model/provider.
     *
     * <p>Initialized during {@link #load(String)} via {@link ModelProvider#createClient(ClassificationProperties)}.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Client client;

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

    @JsonIgnore
    @Neo4jIgnore
    private Evaluation evaluation;

    /**
     * Load classification properties from a YAML file and perform validation and initialization.
     *
     * @param path the file path to the YAML configuration
     * @return an initialized and validated ClassificationProperties instance
     * @throws IOException if the file cannot be read or validation fails
     */
    public static ClassificationProperties load(String path) throws IOException {
        ClassificationProperties properties = loadFromFile(path);
        validateProperties(properties);
        initializeProperties(properties);
        return properties;
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
    private static ClassificationProperties loadFromFile(String path) throws IOException {
        if (path == null || path.isEmpty()) {
            throw new IOException("No path specified for ClassificationProperties file.");
        }
        ObjectMapper mapper = JsonMapper.builder(new YAMLFactory())
                                        .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_ENUMS, true)
                                        .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, false)
                                        .build();
        return mapper.readValue(new File(path), ClassificationProperties.class);
    }

    /**
     * Validate minimal required properties.
     *
     * @param properties the properties instance to validate
     * @throws IOException if mandatory fields (name, version) are missing or empty
     */
    private static void validateProperties(ClassificationProperties properties) throws IOException {
        if (properties.getName() == null || properties.getName().isEmpty()) {
            throw new IOException("Missing name for classification properties");
        }
        if (properties.getVersion() == null || properties.getVersion().isEmpty()) {
            throw new IOException("Missing version for classification properties");
        }
    }

    /**
     * Initialize runtime-only derived fields such as client, pattern, neo4j credentials and taxonomy.
     *
     * @param properties the properties instance to initialize
     * @throws IOException if loading external resources fails (e.g., credentials or taxonomy)
     */
    private static void initializeProperties(ClassificationProperties properties) throws IOException {
        loadClusterNode(properties);
        loadClassificationNode(properties);
        loadModel(properties);
        loadPattern(properties);
        loadNeo4jCredentials(properties);
        loadPrompt(properties);
        properties.setTaxonomy(Taxonomy.load(properties.taxonomyFile));
    }

    /**
     * Compile the regex pattern from {@link PatternConfig} and set {@link #pattern}.
     *
     * <p>If no patternConfig is provided, a default one is created.
     *
     * @param properties the properties instance to update
     */
    private static void loadPattern(ClassificationProperties properties) {
        if (properties.patternConfig == null) {
            properties.patternConfig = new PatternConfig();
        }
        int flags = properties.patternConfig.getFlagsMask();
        properties.setPattern(Pattern.compile(properties.patternConfig.getRegex(), flags));
    }

    /**
     * Create and attach a {@link Classification} node to this properties instance.
     *
     * <p>Only sets the minimal identifying fields (e.g., name).
     *
     * @param properties the properties instance to update
     */
    private static void loadClassificationNode(ClassificationProperties properties) {
        Classification classification = Classification.builder()
                                                      .name(properties.getName())
                                                      .build();
        properties.setClassification(classification);
    }

    /**
     * Create and attach a {@link Cluster} node if {@link #clusterName} is provided.
     *
     * @param properties the properties instance to update
     */
    private static void loadClusterNode(ClassificationProperties properties) {
        if (properties.clusterName == null || properties.clusterName.isEmpty()) {
            return;
        }
        Cluster cluster = Cluster.builder()
                                 .name(properties.clusterName)
                                 .build();
        properties.setCluster(cluster);
    }

    /**
     * Load Neo4j credentials from configured file or use the default credentials.
     *
     * @param properties the properties instance to update
     * @throws IOException if reading credentials file fails
     */
    private static void loadNeo4jCredentials(ClassificationProperties properties) throws IOException {
        if (properties.neo4jCredentialsFile == null || properties.neo4jCredentialsFile.isEmpty()) {
            properties.setNeo4jCredentials(Neo4jCredentials.getDefault());
        } else {
            properties.setNeo4jCredentials(Neo4jCredentials.load(properties.neo4jCredentialsFile));
        }
    }

    /**
     * Parse {@link #modelField}, set {@link #model} and create the appropriate {@link Client}.
     *
     * <p>Expects format "provider:model". Validates provider prefix and delegates client creation.
     *
     * @param properties the properties instance to update
     * @throws IllegalArgumentException if model format is invalid or provider is unknown
     */
    private static void loadModel(ClassificationProperties properties) {
        validateModelFormat(properties.modelField);

        String[] parts = properties.modelField.split(":", 2);
        ModelProvider provider = parseProvider(parts[0]);

        properties.setModel(parts[1].trim());
        properties.setProvider(provider.getName());
        properties.setClient(provider.createClient(properties));
    }

    /**
     * Ensure that the model field contains a provider prefix separated by ':'.
     *
     * @param modelField the raw model field from configuration
     * @throws IllegalArgumentException if format does not contain a colon
     */
    private static void validateModelFormat(String modelField) {
        if (!modelField.contains(":")) {
            throw new IllegalArgumentException(String.format(
                    "Model name must contain a prefix indicating the provider, separated by a colon (:). "
                            + "Allowed providers are: %s",
                    ModelProvider.getAllowedProviders()));
        }
    }

    /**
     * Load the prompt template from a file if {@link #prompt} points to a file.
     *
     * <p>Supports YAML (.yaml, .yml) and text (.txt) files. If the prompt is already
     * a raw string, no action is taken.
     *
     * @param properties the properties instance to update
     */
    private static void loadPrompt(ClassificationProperties properties) {
        if (properties.prompt == null || properties.prompt.isEmpty()) {
            return;
        }
        if (properties.prompt.endsWith(".yaml") || properties.prompt.endsWith(".yml")) {
            try {
                ObjectMapper mapper = JsonMapper.builder(new YAMLFactory())
                                                .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_ENUMS, true)
                                                .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, false)
                                                .build();
                Map<String, String> promptFile = mapper.readValue(new File(properties.prompt), Map.class);
                if (!promptFile.containsKey("prompt")) {
                    throw new IOException("Prompt file does not contain 'prompt' key: " + properties.prompt);
                }
                properties.setPrompt(promptFile.get("prompt"));
                return;
            } catch (IOException e) {
                throw new RuntimeException("Failed to load prompt from file: " + properties.prompt, e);
            }
        }
        if (properties.prompt.endsWith(".txt")) {
            try {
                String promptText = java.nio.file.Files.readString(java.nio.file.Path.of(properties.prompt));
                properties.setPrompt(promptText);
            } catch (IOException e) {
                throw new RuntimeException("Failed to load prompt from file: " + properties.prompt, e);
            }
        }
    }

    /**
     * Parse and validate the provider name into a {@link ModelProvider} enum.
     *
     * @param providerName the raw provider prefix (e.g., "openai")
     * @return the matching {@link ModelProvider}
     * @throws IllegalArgumentException if provider is not supported
     */
    private static ModelProvider parseProvider(String providerName) {
        ModelProvider provider = ModelProvider.fromValue(providerName);
        if (provider == null) {
            throw new IllegalArgumentException(String.format(
                    "Invalid model provider '%s'. Allowed providers are: %s",
                    providerName,
                    ModelProvider.getAllowedProviders()));
        }
        return provider;
    }

    /**
     * Enum of supported model providers and a factory method to create provider-specific clients.
     *
     * <p>Each enum constant contains a human-readable name and a factory function that
     * accepts {@link ClassificationProperties} and returns a {@link Client}.
     */
    @Getter
    @AllArgsConstructor
    private enum ModelProvider {
        ANTHROPIC("anthropic", AnthropicClient::new),
        LOCAL("local", LocalClient::new),
        TOGETHER("together", TogetherClient::new),
        OPENAI("openai", OpenAIClient::new);

        /**
         * Canonical provider name as used in configuration (lowercase).
         */
        private final String name;

        /**
         * Factory used to instantiate provider-specific {@link Client} objects.
         */
        private final ClientFactory factory;

        /**
         * Attempt to map a raw provider string to a {@link ModelProvider} enum.
         *
         * <p>Normalization: uppercase, remove whitespace, hyphens and underscores.
         *
         * @param raw raw provider string (may be null)
         * @return matching ModelProvider or null if none matches
         */
        public static ModelProvider fromValue(String raw) {
            if (raw == null) return null;
            String normalized = raw.toUpperCase(Locale.ROOT)
                                   .replaceAll("[\\s\\-_]", "");
            try {
                return ModelProvider.valueOf(normalized);
            } catch (IllegalArgumentException e) {
                return null;
            }
        }

        /**
         * Return a comma-separated list of allowed provider names for error messages.
         *
         * @return CSV list of provider names
         */
        public static String getAllowedProviders() {
            return Arrays.stream(ModelProvider.values())
                         .map(ModelProvider::getName)
                         .collect(Collectors.joining(", "));
        }

        /**
         * Create a client instance for this provider using the supplied properties.
         *
         * @param properties the classification properties
         * @return a new Client instance for the provider
         */
        public Client createClient(ClassificationProperties properties) {
            return factory.create(properties);
        }

        /**
         * Functional interface defining the factory signature for clients.
         */
        @FunctionalInterface
        private interface ClientFactory {
            /**
             * Create a client using the provided properties.
             *
             * @param properties the classification properties
             * @return a concrete Classifier instance
             */
            Client create(ClassificationProperties properties);
        }
    }

    /**
     * Configuration holder for a regex pattern used to extract labels from client text responses.
     *
     * <p>Contains the regex string and a pipe-separated list of human-friendly flag names which are mapped
     * to {@link Pattern} constants by {@link #getFlagsMask()}.
     */
    @Getter
    @Setter
    @NoArgsConstructor
    @AllArgsConstructor
    private static class PatternConfig {

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
                         .filter(flag -> flag != null)
                         .reduce(0, (a, b) -> a | b);
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

