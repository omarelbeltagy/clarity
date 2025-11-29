package de.tum.claritypipeline.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.credential.BearerTokenCredential;
import com.openai.models.ReasoningEffort;
import com.openai.models.ResponseFormatJsonObject;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.clarityutils.EnvLoader;
import de.tum.clarityutils.SerializationUtils;
import lombok.Getter;
import lombok.Setter;
import okhttp3.MediaType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * TogetherClient is a concrete implementation of the Client interface that
 * communicates with the Together.ai HTTP API using OkHttp.
 * <p>
 * It builds requests according to the provided ModelConfig and extracts either
 * plain text or structured JSON object responses depending on the configured
 * ResponseFormat.
 */
@Getter
@Setter
public class TogetherClient implements Client {
    /**
     * Base URL for Together chat completions endpoint.
     */
    private static final String TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions";

    /**
     * MediaType for JSON requests.
     */
    private static final MediaType JSON_MEDIA_TYPE = MediaType.get("application/json; charset=utf-8");

    /**
     * Pattern to detect markdown JSON code blocks (```json ... ```).
     */
    private static final Pattern MARKDOWN_CODE_BLOCK = Pattern.compile("```(?:json)?\\s*(.+?)```", Pattern.DOTALL);

    /**
     * Logger for this client.
     */
    private final Logger log = LoggerFactory.getLogger(TogetherClient.class);

    /**
     * Configuration for the model to use (name, tokens, temperature, etc.).
     */
    private final ModelProperties properties;

    /**
     * Client for API Calls
     */
    private final OpenAIClient client;

    /**
     * Thread-local ObjectMapper to avoid creating multiple mappers across threads.
     */
    private final ThreadLocal<ObjectMapper> threadLocalMapper;


    /**
     * Construct a TogetherClient with the given ModelConfig.
     * <p>
     * The constructor validates the TOGETHER_API_KEY environment variable,
     * builds an OkHttpClient and prepares a thread-local ObjectMapper.
     *
     * @param properties model configuration to use for requests
     */
    public TogetherClient(ModelProperties properties) {
        String apiKey = EnvLoader.get("TOGETHER_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            throw new IllegalStateException(
                    "TOGETHER_API_KEY environment variable is not set. Please set it to use OpenAIClassifier.");
        }
        this.threadLocalMapper = ThreadLocal.withInitial(ObjectMapper::new);
        this.properties = properties;
        this.client = OpenAIOkHttpClient.builder()
                                        .credential(BearerTokenCredential.create(apiKey))
                                        .baseUrl("https://api.together.xyz/v1/") // Together AI Endpoint
                                        .build();
    }

    /**
     * Make a request to the Together API and return a plain text response.
     *
     * @param prompt the input prompt to send to the model
     * @return the textual response or null on failure
     */
    @Override
    public String makeRequest(String prompt) {
        return makeRequest(prompt, String.class);
    }

    /**
     * Make a request to the Together API and parse the response into the given class.
     * <p>
     * If the configured response format is not JSON_OBJECT, only String.class is supported.
     *
     * @param prompt the input prompt
     * @param clazz  the expected result class
     * @param <T>    type of the expected result
     * @return parsed response of type T or null on error
     */
    @Override
    public <T> T makeRequest(String prompt, Class<T> clazz) {
        if (properties.getResponseFormat() != ModelProperties.ResponseFormat.JSON_OBJECT && clazz != String.class) {
            throw new IllegalArgumentException(
                    "Unsupported class type for text response: " + clazz.getName()
                            + ". Only String is supported for Response Format: "
                            + properties.getResponseFormat());
        }

        return extractStructuredResponse(handleRequest(prompt, clazz), clazz);
    }

    private <T> String handleRequest(String prompt, Class<T> clazz) {
        ChatCompletionCreateParams.Builder paramsBuilder = ChatCompletionCreateParams.builder()
                                                                                     .model(properties.getName())
                                                                                     .maxCompletionTokens(
                                                                                             properties.getMaxTokens())
                                                                                     .addUserMessage(prompt);
        if (properties.getResponseFormat() == ModelProperties.ResponseFormat.JSON_OBJECT) {
            paramsBuilder.responseFormat(ResponseFormatJsonObject.builder().build());
        }
        if (properties.getTemperature() != null) {
            paramsBuilder.temperature(properties.getTemperature());
        }
        if (properties.getTopP() != null) {
            paramsBuilder.topP(properties.getTopP());
        }
        if (properties.getReasoningEffort() != null) {
            paramsBuilder.reasoningEffort(ReasoningEffort.of(properties.getReasoningEffort()));
        }

        return client.chat().completions().create(paramsBuilder.build()).choices()
                     .getFirst()
                     .message()
                     .content()
                     .orElse(null);
    }

    /**
     * Extract the relevant content from the raw response body and parse it into the given class.
     *
     * @param responseBody raw HTTP response body as string
     * @param clazz        expected result class
     * @param <T>          type of the expected result
     * @return parsed object of type T or null on failure
     */
    private <T> T extractStructuredResponse(String responseBody, Class<T> clazz) {
        try {
            if (responseBody == null) {
                return null;
            }
            return parseContent(responseBody, clazz);
        } catch (Exception e) {
            log.error("Failed to extract structured response from Together API", e);
            return null;
        }
    }

    /**
     * Parse the content string into the expected class. Handles label-only (String)
     * and JSON parsing for structured responses, including arrays and markdown cleanup.
     *
     * @param content raw content extracted from the response
     * @param clazz   expected result class
     * @param <T>     type of the expected result
     * @return parsed object of type T or null on failure
     * @throws IOException if JSON parsing fails
     */
    private <T> T parseContent(String content, Class<T> clazz) throws IOException {
        if (clazz == String.class) {
            String labelResult = parseLabelFormat(content);
            return clazz.cast(labelResult);
        }

        String cleanedContent = removeMarkdownCodeBlocks(content);
        return parseJsonContent(cleanedContent, clazz);
    }

    /**
     * Extract a simple label from the content using the configured pattern.
     * If the pattern does not match, the original content is returned.
     *
     * @param content content to parse for a label
     * @return matched group 1 trimmed or original content
     */
    private String parseLabelFormat(String content) {
        Matcher matcher = properties.getPattern().matcher(content);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        return content;
    }

    /**
     * Remove markdown JSON code block wrappers (```json ... ```) if present.
     *
     * @param content content that may contain markdown code blocks
     * @return inner content without markdown fences, trimmed
     */
    private String removeMarkdownCodeBlocks(String content) {
        Matcher matcher = MARKDOWN_CODE_BLOCK.matcher(content);
        return matcher.find() ? matcher.group(1).trim() : content;
    }

    /**
     * Parse JSON content into the expected class. Cleans up trailing commas and
     * handles both object and array responses.
     *
     * @param content JSON content string
     * @param clazz   expected class
     * @param <T>     type parameter
     * @return deserialized object of type T or null if parsing fails
     * @throws IOException if JSON parsing fails
     */
    private <T> T parseJsonContent(String content, Class<T> clazz) throws IOException {
        content = content.replaceAll(",\\s*([}\\]])", "$1").replaceAll(",\\s*\"\\s*}", "}");
        ObjectMapper objectMapper = threadLocalMapper.get();
        JsonNode parsedNode = objectMapper.readTree(content);

        if (parsedNode.isArray() && !parsedNode.isEmpty()) {
            return parseArrayResponse(parsedNode, clazz);
        }

        return SerializationUtils.deserialize(parsedNode.toString(), clazz);
    }

    /**
     * Iterate over elements of a JSON array and try deserializing each element
     * into the expected class until one succeeds.
     *
     * @param arrayNode JSON array node
     * @param clazz     expected class
     * @param <T>       type parameter
     * @return first successfully deserialized element or null if none match
     */
    private <T> T parseArrayResponse(JsonNode arrayNode, Class<T> clazz) {
        for (JsonNode element : arrayNode) {
            try {
                T result = SerializationUtils.deserialize(
                        element.toString(),
                        clazz
                );
                if (result != null) {
                    return result;
                }
            } catch (Exception ignore) {}
        }
        return null;
    }
}