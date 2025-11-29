package de.tum.claritypipeline.client;

import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.messages.ContentBlock;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.MessageCreateParams;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.clarityutils.EnvLoader;
import de.tum.clarityutils.SerializationUtils;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * AnthropicClient is a concrete implementation of the Client interface
 * that communicates with Anthropic models using the Anthropic SDK.
 * <p>
 * It supports both text-based responses and structured JSON object responses,
 * depending on the configured ModelConfig.
 */
@Getter
@Setter
public class AnthropicClient implements Client {
    /**
     * Logger instance for logging errors and debug information.
     */
    private final Logger log = org.slf4j.LoggerFactory.getLogger(AnthropicClient.class);

    /**
     * Configuration properties for the model (name, tokens, temperature, pattern, etc.).
     */
    private final ModelProperties properties;

    /**
     * Underlying Anthropic SDK client used to send requests.
     */
    private final com.anthropic.client.AnthropicClient client;

    /**
     * ObjectMapper instance for JSON processing (if needed).
     */
    private final ObjectMapper objectMapper;

    /**
     * Construct an AnthropicClient using the provided model configuration.
     * The constructor reads ANTHROPIC_API_KEY from the environment and initializes the SDK client.
     *
     * @param properties model configuration containing model name and runtime settings
     * @throws IllegalStateException if ANTHROPIC_API_KEY is not set
     */
    public AnthropicClient(ModelProperties properties) {
        String apiKey = EnvLoader.get("ANTHROPIC_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            throw new IllegalStateException(
                    "ANTHROPIC_API_KEY environment variable is not set. Please set it to use AnthropicClassifier.");
        }
        this.properties = properties;
        this.client = AnthropicOkHttpClient.builder()
                                           .apiKey(apiKey)
                                           .build();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Send a request to the Anthropic model and return a plain text response.
     *
     * @param prompt prompt text to send to the model
     * @return model output as String, or null on error
     */
    @Override
    public String makeRequest(String prompt) {
        return makeRequest(prompt, String.class);
    }

    /**
     * Send a request to the Anthropic model and parse the response into the specified type.
     * For ResponseFormat.JSON_OBJECT the response is deserialized into clazz, otherwise only String is supported.
     *
     * @param prompt the prompt to send
     * @param clazz  desired response class
     * @param <T>    response type
     * @return parsed response instance or null on error
     */
    @Override
    public <T> T makeRequest(String prompt, Class<T> clazz) {
        try {
            if (properties.getResponseFormat() != ModelProperties.ResponseFormat.JSON_OBJECT && clazz != String.class) {
                throw new IllegalArgumentException(
                        "Unsupported class type for text response: " + clazz.getName()
                                + ". Only String is supported for Response Format: "
                                + properties.getResponseFormat());
            }
            MessageCreateParams.Builder paramsBuilder = MessageCreateParams.builder()
                                                                           .model(properties.getName())
                                                                           .maxTokens(properties.getMaxTokens())
                                                                           .addUserMessage(prompt);
            if (properties.getTemperature() != null) {
                paramsBuilder.temperature(properties.getTemperature());
            }

            Message response = client.messages().create(paramsBuilder.build());
            if (response.content().isEmpty()) {
                log.error("Received empty response from Anthropic for prompt: {}", prompt);
                return null;
            }

            T result;
            if (properties.getResponseFormat() == ModelProperties.ResponseFormat.JSON_OBJECT) {
                result = parseStructuredResponse(response, clazz);
            } else {
                result = clazz.cast(parseTextResponse(response));
            }

            if (result == null) {
                log.error("Received invalid classification response from Anthropic for prompt: {}", prompt);
            }
            return result;
        } catch (Exception e) {
            log.error("Error during Anthropic classification request: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Try to extract a JSON payload from an Anthropic SDK JSON wrapper string.
     * The method attempts to find an inner JSON object represented after "text=".
     *
     * @param input raw input string possibly containing inner JSON
     * @return extracted JSON string or null if none found
     */
    private String extractInnerJson(String input) {
        final Pattern pattern =
                Pattern.compile("text=\\{(.*)}\\s*}", Pattern.DOTALL);

        if (input == null || input.isBlank()) return null;

        Matcher matcher = pattern.matcher(input);
        if (matcher.find()) {
            String json = matcher.group(1).trim();

            if (!json.startsWith("{")) json = "{" + json;
            if (!json.endsWith("}")) json = json + "}";

            return json;
        }
        return null;
    }

    /**
     * Parse a structured JSON response from an Anthropic Message into the provided class.
     * The method inspects ContentBlock entries, handles embedded JSON and fenced ```json blocks.
     *
     * @param response the Anthropic message response
     * @param clazz    class to deserialize into
     * @param <T>      type parameter
     * @return deserialized instance of clazz, or null if parsing fails
     */
    private <T> T parseStructuredResponse(Message response, Class<T> clazz) {
        try {
            for (ContentBlock block : response.content()) {
                if (block.isText()) {
                    if (block._json().isPresent()) {
                        String jsonString = block._json().get().toString();
                        String jsonPayload = extractInnerJson(jsonString);
                        if (jsonPayload != null) {
                            try {
                                return SerializationUtils.deserialize(
                                        jsonPayload, clazz);
                            } catch (Exception ignore) {}
                            jsonString = jsonPayload;
                        }
                        if (jsonString.contains("```json")) {
                            int startIndex = jsonString.indexOf("```json") + "```json".length();
                            String trimmed = jsonString.substring(startIndex).trim();
                            int endIndex = trimmed.indexOf("```");
                            if (endIndex > 0) {
                                jsonString = trimmed.substring(0, endIndex).trim();
                                return SerializationUtils.deserialize(
                                        jsonString, clazz);
                            }
                            endIndex = jsonString.indexOf("```", startIndex);
                            if (endIndex > startIndex) {
                                jsonString = jsonString.substring(startIndex, endIndex).trim();
                                return SerializationUtils.deserialize(
                                        jsonString, clazz);
                            }
                        }
                    }
                }
            }
        } catch (Exception ignore) {}
        return null;
    }

    /**
     * Parse a plain text response from an Anthropic Message using the configured regex pattern.
     * If a match is found the first capturing group is returned.
     *
     * @param response Anthropic message response
     * @return matched text or null if nothing matched
     */
    private String parseTextResponse(Message response) {
        try {
            for (ContentBlock block : response.content()) {
                if (block.isText()) {
                    if (block.text().isPresent()) {
                        String content = block.text().get().toString();
                        Matcher matcher = properties.getPattern().matcher(content);
                        if (matcher.find()) {
                            return matcher.group(1).trim();
                        }
                    }
                }
            }
        } catch (Exception ignore) {}
        return null;
    }
}