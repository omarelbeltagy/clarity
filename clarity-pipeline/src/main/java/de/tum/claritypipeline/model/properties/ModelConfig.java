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
import de.tum.claritypipeline.client.Client;
import de.tum.clarityutils.AfterDeserialization;
import lombok.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Map;
import java.util.regex.Pattern;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class ModelConfig {

    /**
     * The model name
     */
    @JsonProperty("name")
    private String name;

    /**
     * The model provider name (e.g., "openai", "anthropic").
     */
    @JsonProperty("provider")
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
    @Neo4jIgnore
    @Setter(AccessLevel.NONE)
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
     * <p>Value is between 0 and 1. Default is 0.9.
     */
    @JsonProperty("top-p")
    @JsonPropertyDescription("The nucleus sampling parameter for the language model.")
    private double topP = 0.9;

    /**
     * Temperature parameter for the language model.
     *
     * <p>Higher values produce more creative outputs. Default is 1.0.
     */
    @JsonProperty("temperature")
    @JsonPropertyDescription("The temperature setting for the language model.")
    private double temperature = 1.0;

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
     * Configuration for the pattern used to extract labels from textual responses.
     *
     * <p>Contains the regex and flags; defaults are provided by {@link PatternConfig}.
     */
    @JsonProperty("pattern")
    @JsonPropertyDescription("The pattern configuration for extracting labels from text. Contains the regex and flags.")
    @Neo4jIgnore
    @Setter(AccessLevel.NONE)
    @Getter(AccessLevel.NONE)
    private PatternConfig patternConfig;

    /**
     * Whether to inject response format instructions into the prompt.
     *
     * <p>Default is true.
     */
    @JsonProperty("inject-response-format")
    @JsonPropertyDescription("Whether to inject response format instructions into the prompt.")
    private boolean injectResponseFormat = true;

    @JsonProperty("raq")
    private RaqProperties raqProperties = new RaqProperties();

    /**
     * Compiled regex pattern for label extraction from text responses.
     *
     * <p>Initialized after deserialization.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Pattern pattern;

    /**
     * Sets the response format and adjusts structured output accordingly.
     *
     * <p>If the response format is TEXT, structured output is set to false.
     *
     * @param responseFormat The desired response format.
     */
    @JsonSetter("response-format")
    public void setResponseFormat(ResponseFormat responseFormat) {
        this.responseFormat = responseFormat;
        if (responseFormat == ResponseFormat.TEXT) {
            this.structuredOutput = false;
        }
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
            } catch (IOException e) {
                throw new RuntimeException("Failed to load prompt from file: " + prompt, e);
            }
        }
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
        if (patternConfig == null) {
            patternConfig = new PatternConfig();
        }
        int flags = patternConfig.getFlagsMask();
        this.pattern = Pattern.compile(patternConfig.getRegex(), flags);
        this.client = Client.create(this);
    }
}
