package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonProperty;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.ClassificationRequest;
import de.tum.claritypipeline.model.ClassificationResult;
import de.tum.claritypipeline.model.ResponseFormat;
import de.tum.claritypipeline.model.properties.ModelConfig;
import de.tum.claritypipeline.utils.PromptUtils;
import de.tum.clarityutils.SerializationUtils;
import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class SingleStrategy implements ClassificationStrategy {
    @JsonProperty("model")
    private ModelConfig model;

    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        String prompt = switch (model.getClient()) {
            case LocalClient ignore -> SerializationUtils.serialize(request);
            default -> PromptUtils.replacePrompt(request, model.getPrompt(), model.getResponseFormat(),
                                                 model.isInjectResponseFormat(), request.getTaxonomy());
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
