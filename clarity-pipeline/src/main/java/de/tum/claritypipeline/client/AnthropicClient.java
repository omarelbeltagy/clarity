package de.tum.claritypipeline.client;

import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.messages.ContentBlock;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.MessageCreateParams;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.tum.claritypipeline.model.ResponseFormat;
import de.tum.claritypipeline.model.properties.ModelConfig;
import de.tum.clarityutils.EnvLoader;
import de.tum.clarityutils.SerializationUtils;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Getter
@Setter
public class AnthropicClient implements Client {
    private final Logger log = org.slf4j.LoggerFactory.getLogger(AnthropicClient.class);

    private final ModelConfig properties;
    private final com.anthropic.client.AnthropicClient client;
    private final ObjectMapper objectMapper;

    public AnthropicClient(ModelConfig properties) {
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
            MessageCreateParams params = MessageCreateParams.builder()
                                                            .model(properties.getName())
                                                            .maxTokens(properties.getMaxTokens())
                                                            .temperature(properties.getTemperature())
                                                            .addUserMessage(prompt)
                                                            .build();

            Message response = client.messages().create(params);
            if (response.content().isEmpty()) {
                log.error("Received empty response from Anthropic for prompt: {}", prompt);
                return null;
            }

            T result;
            if (properties.getResponseFormat() == ResponseFormat.JSON_OBJECT) {
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