package de.tum.claritypipeline.client;

import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.credential.BearerTokenCredential;
import com.openai.models.ChatModel;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.StructuredChatCompletionCreateParams;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.model.config.ResponseFormat;
import de.tum.clarityutils.EnvLoader;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;

import java.util.regex.Matcher;

/**
 * OpenAIClient communicates with OpenAI chat models using the OpenAI SDK.
 * It supports text responses and structured response deserialization when supported by the model.
 */
@Getter
@Setter
public class OpenAIClient implements Client {
    /**
     * Logger for OpenAIClient.
     */
    private final Logger log = org.slf4j.LoggerFactory.getLogger(OpenAIClient.class);

    /**
     * Model configuration (name, tokens, temperature, etc.).
     */
    private final ModelProperties properties;

    /**
     * ChatModel wrapper representing the chosen model in the SDK.
     */
    private final ChatModel chatModel;

    /**
     * Underlying OpenAI SDK client.
     */
    private final com.openai.client.OpenAIClient client;

    /**
     * Create a new OpenAIClient using the environment OPENAI_API_KEY and the provided configuration.
     *
     * @param properties model configuration
     * @throws IllegalStateException if OPENAI_API_KEY is missing
     */
    public OpenAIClient(ModelProperties properties) {
        String apiKey = EnvLoader.get("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            throw new IllegalStateException(
                    "OPENAI_API_KEY environment variable is not set. Please set it to use OpenAIClassifier.");
        }
        this.chatModel = ChatModel.of(properties.getName());
        this.properties = properties;
        this.client = new OpenAIOkHttpClient.Builder().credential(
                BearerTokenCredential.create(apiKey)).build();
    }

    /**
     * Send a prompt and return a plain text response.
     *
     * @param prompt prompt to send
     * @return response text or null
     */
    @Override
    public String makeRequest(String prompt) {
        return makeRequest(prompt, String.class);
    }

    /**
     * Send a prompt and parse result either as structured object or text depending on configuration.
     *
     * @param prompt input prompt
     * @param clazz  expected return type
     * @param <T>    response type
     * @return parsed response or null on error
     */
    @Override
    public <T> T makeRequest(String prompt, Class<T> clazz) {
        try {
            T result;
            if (properties.getResponseFormat() == ResponseFormat.JSON_OBJECT) {
                result = makeStructuredRequest(prompt, clazz);
            } else {
                if (clazz == String.class) {
                    result = clazz.cast(makeTextRequest(prompt));
                } else {
                    throw new IllegalArgumentException(
                            "Unsupported class type for text response: " + clazz.getName()
                                    + ". Only String is supported for Response Format: "
                                    + properties.getResponseFormat());
                }
            }
            if (result == null) {
                log.error("Received no classification response from OpenAI for prompt: {}", prompt);
            }
            return result;
        } catch (Exception e) {
            log.error("Error during OpenAI classification request: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Build and send a structured request to OpenAI and return a deserialized object.
     *
     * @param prompt prompt text
     * @param clazz  expected class
     * @param <T>    response type
     * @return deserialized response or null
     */
    private <T> T makeStructuredRequest(String prompt, Class<T> clazz) {
        StructuredChatCompletionCreateParams<T> params;
        if (properties.getName().toLowerCase().startsWith("gpt-4")) {
            params =
                    ChatCompletionCreateParams.builder()
                                              .model(chatModel)
                                              .temperature(properties.getTemperature())
                                              .topP(properties.getTopP())
                                              .maxCompletionTokens(properties.getMaxTokens())
                                              .addUserMessage(
                                                      prompt)
                                              .responseFormat(clazz)
                                              .build();
        } else {
            params =
                    ChatCompletionCreateParams.builder()
                                              .model(chatModel)
                                              .maxCompletionTokens(properties.getMaxTokens())
                                              .addUserMessage(
                                                      prompt)
                                              .responseFormat(clazz)
                                              .build();
        }

        return client.chat().completions().create(params).choices()
                     .getFirst()
                     .message()
                     .content()
                     .orElse(null);
    }

    /**
     * Send a text-only request and return the parsed label using the configured regex.
     *
     * @param prompt prompt text
     * @return extracted label string or null
     */
    private String makeTextRequest(String prompt) {
        ChatCompletionCreateParams params =
                ChatCompletionCreateParams.builder()
                                          .model(chatModel)
                                          .temperature(properties.getTemperature())
                                          .topP(properties.getTopP())
                                          .maxCompletionTokens(properties.getMaxTokens())
                                          .addUserMessage(
                                                  prompt)
                                          .build();

        String content = client.chat().completions().create(params).choices()
                               .getFirst()
                               .message()
                               .content()
                               .orElse(null);

        return parseResponseText(content);
    }

    /**
     * Parse raw content returned from OpenAI using the configured regex pattern and return the first group.
     *
     * @param content raw model content
     * @return matched group or null
     */
    private String parseResponseText(String content) {
        Matcher matcher = properties.getPattern().matcher(content);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        return null;
    }
}
