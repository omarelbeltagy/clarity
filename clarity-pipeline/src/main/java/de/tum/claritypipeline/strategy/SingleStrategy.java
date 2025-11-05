package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.ClassificationRequest;
import de.tum.claritypipeline.model.ClassificationResult;
import de.tum.claritypipeline.model.properties.ModelConfig;
import de.tum.claritypipeline.model.properties.ResponseFormat;
import de.tum.claritypipeline.utils.PromptUtils;
import de.tum.clarityutils.SerializationUtils;
import lombok.*;

/**
 * Strategy that performs a single-model classification call.
 * <p>
 * Depending on the configured ModelConfig and client type the strategy either:
 * - serializes the request and sends it directly to a LocalClient, or
 * - builds a prompt using PromptUtils and sends it to a remote model client.
 * <p>
 * The strategy supports both structured JSON responses (ResponseFormat.JSON_OBJECT)
 * which are deserialized into a ClassificationResult, and plain text responses
 * which are interpreted as the label name.
 */
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class SingleStrategy implements ClassificationStrategy {
    /**
     * Model configuration to use for classification.
     *
     * This configuration supplies the client, prompt template, response format
     * and other parameters required to execute a single classification.
     */
    @JsonProperty("model")
    @JsonPropertyDescription("The model configuration to use for classification.")
    private ModelConfig model;

    /**
     * Execute the single-call classification.
     *
     * Behavior:
     * - If the configured client is a LocalClient, the method serializes the
     *   entire ClassificationRequest and hands it to the client.
     * - Otherwise, it calls PromptUtils.replacePrompt to produce the prompt
     *   string to send to the remote client.
     * - If the model expects JSON_OBJECT output the method deserializes the
     *   response into a ClassificationResult. For non-JSON responses the raw
     *   string is used as the predicted label name.
     *
     * @param request the classification request containing text and taxonomy.
     * @return a ClassificationResult representing the predicted class and any
     *         available metadata (explanation, confidence). For plain text
     *         responses the result will have only the name field populated.
     */
    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        String prompt = switch (model.getClient()) {
            case LocalClient ignore -> SerializationUtils.serialize(request);
            default -> PromptUtils.replacePrompt(request, model.getPrompt(), model.getResponseFormat(),
                                                 model.isInjectResponseFormat(), request.getTaxonomy(),
                                                 model.getRaqProperties(), ClassificationResult.class);
        };

        if (model.getResponseFormat() == ResponseFormat.JSON_OBJECT) {
            return model.getClient()
                        .makeRequest(prompt, ClassificationResult.class);
        } else {
            String response = model.getClient()
                                   .makeRequest(prompt);
            return ClassificationResult.builder().name(response).build();
        }
    }
}
