package de.tum.claritypipeline.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import de.tum.claritypipeline.model.ClassificationResult;
import de.tum.claritypipeline.model.ResponseFormat;
import de.tum.claritypipeline.model.properties.ModelConfig;
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

@Getter
@Setter
public class TogetherClient implements Client {
    private static final String TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions";
    private static final MediaType JSON_MEDIA_TYPE = MediaType.get("application/json; charset=utf-8");
    private static final Pattern MARKDOWN_CODE_BLOCK = Pattern.compile("```(?:json)?\\s*(.+?)```", Pattern.DOTALL);

    private final Logger log = LoggerFactory.getLogger(TogetherClient.class);
    private final ModelConfig properties;
    private final OkHttpClient httpClient;
    private final String apiKey;
    private final ThreadLocal<ObjectMapper> threadLocalMapper;


    public TogetherClient(ModelConfig properties) {
        this.apiKey = validateApiKey();
        this.properties = properties;
        this.httpClient = buildHttpClient();
        this.threadLocalMapper = ThreadLocal.withInitial(ObjectMapper::new);
    }

    private String validateApiKey() {
        String key = EnvLoader.get("TOGETHER_API_KEY");
        if (key == null || key.isBlank()) {
            throw new IllegalStateException(
                    "TOGETHER_API_KEY environment variable is not set. Please set it to use LlamaClassifier.");
        }
        return key;
    }

    private OkHttpClient buildHttpClient() {
        ConnectionPool pool = new ConnectionPool(20, 5, TimeUnit.MINUTES);
        return new OkHttpClient.Builder()
                .connectionPool(pool)
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();
    }

    @Override
    public String makeRequest(String prompt) {
        return makeRequest(prompt, String.class);
    }

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

    private Request buildHttpRequest(String requestBody) {
        return new Request.Builder()
                .url(TOGETHER_API_URL)
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(requestBody, JSON_MEDIA_TYPE))
                .build();
    }

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

    private void logFailedResponse(Response response) throws IOException {
        String errorBody = response.body() != null ? response.body().string() : "No body";
        log.error("Together API request failed with code {}: {}", response.code(), errorBody);
    }

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

    private void addJsonSchema(ObjectNode responseFormatNode) throws IOException {
        JsonScheme<ClassificationResult> jsonScheme = new JsonScheme<>(ClassificationResult.class);
        ObjectMapper objectMapper = threadLocalMapper.get();
        ObjectNode schemeNode = objectMapper.createObjectNode();
        schemeNode.set("properties", objectMapper.readTree(jsonScheme.getPropertiesString()));
        responseFormatNode.set("schema", schemeNode);
    }

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

    private <T> T parseContent(String content, Class<T> clazz) throws IOException {
        if (clazz == String.class) {
            String labelResult = parseLabelFormat(content);
            return clazz.cast(labelResult);
        }

        String cleanedContent = removeMarkdownCodeBlocks(content);
        return parseJsonContent(cleanedContent, clazz);
    }

    private String parseLabelFormat(String content) {
        Matcher matcher = properties.getPattern().matcher(content);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        return content;
    }

    private String removeMarkdownCodeBlocks(String content) {
        Matcher matcher = MARKDOWN_CODE_BLOCK.matcher(content);
        return matcher.find() ? matcher.group(1).trim() : content;
    }

    private <T> T parseJsonContent(String content, Class<T> clazz) throws IOException {
        ObjectMapper objectMapper = threadLocalMapper.get();
        JsonNode parsedNode = objectMapper.readTree(content);

        if (parsedNode.isArray() && !parsedNode.isEmpty()) {
            return parseArrayResponse(parsedNode, clazz);
        }

        return SerializationUtils.deserialize(parsedNode.toString(), clazz);
    }

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