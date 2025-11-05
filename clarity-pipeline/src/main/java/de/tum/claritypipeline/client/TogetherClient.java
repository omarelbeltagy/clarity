package de.tum.claritypipeline.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import de.tum.claritypipeline.model.ClassificationResult;
import de.tum.claritypipeline.model.properties.ModelConfig;
import de.tum.claritypipeline.model.properties.ResponseFormat;
import de.tum.clarityutils.EnvLoader;
import de.tum.clarityutils.JsonScheme;
import de.tum.clarityutils.SerializationUtils;
import lombok.Getter;
import lombok.Setter;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;
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
    private final ModelConfig properties;

    /**
     * OkHttp client used to perform HTTP calls.
     */
    private final OkHttpClient httpClient;

    /**
     * API key used for authorization with Together API.
     */
    private final String apiKey;

    /**
     * Thread-local ObjectMapper to avoid creating multiple mappers across threads.
     */
    private final ThreadLocal<ObjectMapper> threadLocalMapper;


    /**
     * Construct a TogetherClient with the given ModelConfig.
     *
     * The constructor validates the TOGETHER_API_KEY environment variable,
     * builds an OkHttpClient and prepares a thread-local ObjectMapper.
     *
     * @param properties model configuration to use for requests
     */
    public TogetherClient(ModelConfig properties) {
        this.apiKey = validateApiKey();
        this.properties = properties;
        this.httpClient = buildHttpClient();
        this.threadLocalMapper = ThreadLocal.withInitial(ObjectMapper::new);
    }

    /**
     * Validate and return the Together API key from environment variables.
     *
     * @return the API key string
     * @throws IllegalStateException if the TOGETHER_API_KEY is missing or blank
     */
    private String validateApiKey() {
        String key = EnvLoader.get("TOGETHER_API_KEY");
        if (key == null || key.isBlank()) {
            throw new IllegalStateException(
                    "TOGETHER_API_KEY environment variable is not set. Please set it to use LlamaClassifier.");
        }
        return key;
    }

    /**
     * Build and configure the OkHttpClient used for requests.
     *
     * @return configured OkHttpClient
     */
    private OkHttpClient buildHttpClient() {
        ConnectionPool pool = new ConnectionPool(20, 5, TimeUnit.MINUTES);
        return new OkHttpClient.Builder()
                .connectionPool(pool)
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
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
     *
     * If the configured response format is not JSON_OBJECT, only String.class is supported.
     *
     * @param prompt the input prompt
     * @param clazz  the expected result class
     * @param <T>    type of the expected result
     * @return parsed response of type T or null on error
     */
    @Override
    public <T> T makeRequest(String prompt, Class<T> clazz) {
        try {
            if (properties.getResponseFormat() != ResponseFormat.JSON_OBJECT && clazz != String.class) {
                throw new IllegalArgumentException(
                        "Unsupported class type for text response: " + clazz.getName()
                                + ". Only String is supported for Response Format: "
                                + properties.getResponseFormat());
            }
            String requestBody = buildRequestBody(prompt);
            Request request = buildHttpRequest(requestBody);

            try (Response response = httpClient.newCall(request).execute()) {
                return handleResponse(response, clazz);
            }
        } catch (IOException e) {
            log.error("Together API request failed", e);
            return null;
        }
    }

    /**
     * Build the OkHttp Request object for the provided JSON request body.
     *
     * @param requestBody serialized JSON request body
     * @return OkHttp Request ready to be executed
     */
    private Request buildHttpRequest(String requestBody) {
        return new Request.Builder()
                .url(TOGETHER_API_URL)
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(requestBody, JSON_MEDIA_TYPE))
                .build();
    }

    /**
     * Handle the HTTP response, checking success and parsing the body.
     *
     * @param response OkHttp Response object
     * @param clazz    expected result class
     * @param <T>      type of the expected result
     * @return parsed result of type T or null on failure
     * @throws IOException if reading the response body fails
     */
    private <T> T handleResponse(Response response, Class<T> clazz) throws IOException {
        if (!response.isSuccessful()) {
            logFailedResponse(response);
            return null;
        }

        ResponseBody body = response.body();
        if (body == null) {
            log.error("Empty response body from Together API");
            return null;
        }

        return extractStructuredResponse(body.string(), clazz);
    }

    /**
     * Log details about a failed HTTP response.
     *
     * @param response the failed response
     * @throws IOException if reading the error body fails
     */
    private void logFailedResponse(Response response) throws IOException {
        String errorBody = response.body() != null ? response.body().string() : "No body";
        log.error("Together API request failed with code {}: {}", response.code(), errorBody);
    }

    /**
     * Build the JSON request body according to the ModelConfig and the provided prompt.
     *
     * @param prompt the user prompt content
     * @return serialized JSON string for the request body
     * @throws IOException if serialization fails
     */
    private String buildRequestBody(String prompt) throws IOException {
        ObjectMapper objectMapper = threadLocalMapper.get();
        ObjectNode node = objectMapper.createObjectNode();
        node.put("model", properties.getName());
        node.put("max_tokens", properties.getMaxTokens());
        node.put("temperature", properties.getTemperature());
        node.put("top_p", properties.getTopP());

        ObjectNode messageNode = objectMapper.createObjectNode();
        messageNode.put("role", "user");
        messageNode.put("content", prompt);
        node.set("messages", objectMapper.createArrayNode().add(messageNode));

        addResponseFormat(node);

        return objectMapper.writeValueAsString(node);
    }

    /**
     * Add the response_format object to the request when JSON_OBJECT is requested.
     *
     * @param node root request object node to modify
     * @throws IOException if schema serialization fails
     */
    private void addResponseFormat(ObjectNode node) throws IOException {
        if (properties.getResponseFormat() != ResponseFormat.JSON_OBJECT) {
            return;
        }

        ObjectMapper objectMapper = threadLocalMapper.get();
        ObjectNode responseFormatNode = objectMapper.createObjectNode();
        responseFormatNode.put("type", "json_object");

        if (properties.isStructuredOutput()) {
            addJsonSchema(responseFormatNode);
        }

        node.set("response_format", responseFormatNode);
    }

    /**
     * Add a JSON schema for the expected structured output based on ClassificationResult.
     *
     * @param responseFormatNode node where the schema should be attached
     * @throws IOException if building the schema nodes fails
     */
    private void addJsonSchema(ObjectNode responseFormatNode) throws IOException {
        JsonScheme<ClassificationResult> jsonScheme = new JsonScheme<>(ClassificationResult.class);
        ObjectMapper objectMapper = threadLocalMapper.get();
        ObjectNode schemeNode = objectMapper.createObjectNode();
        schemeNode.set("properties", objectMapper.readTree(jsonScheme.getPropertiesString()));
        responseFormatNode.set("schema", schemeNode);
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
            String content = extractContentFromResponse(responseBody);
            if (content == null) {
                return null;
            }

            return parseContent(content, clazz);
        } catch (Exception e) {
            log.error("Failed to extract structured response from Together API", e);
            return null;
        }
    }

    /**
     * Read the JSON response and obtain the message content from the first choice.
     *
     * @param responseBody raw response body
     * @return content string, trimmed, or null if not found
     * @throws IOException if JSON parsing fails
     */
    private String extractContentFromResponse(String responseBody) throws IOException {
        ObjectMapper objectMapper = threadLocalMapper.get();
        JsonNode rootNode = objectMapper.readTree(responseBody);
        JsonNode choicesNode = rootNode.path("choices");

        if (!choicesNode.isArray() || choicesNode.isEmpty()) {
            log.error("No choices found in Together API response");
            return null;
        }

        String content = choicesNode.get(0).path("message").path("content").asText();

        if (content == null || content.isBlank()) {
            log.error("No content found in Together API response");
            return null;
        }

        return content.trim();
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