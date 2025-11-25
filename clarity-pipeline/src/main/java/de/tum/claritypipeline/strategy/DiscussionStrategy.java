package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.model.config.ResponseFormat;
import de.tum.claritypipeline.utils.PromptUtils;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class DiscussionStrategy implements ClassificationStrategy {
    @JsonIgnore
    private static final String TARGET_LABEL_PLACEHOLDER = "{target_label}";
    @JsonIgnore
    private static final String REASONS_PLACEHOLDER = "{reasons}";
    @JsonProperty("discussion-model")
    @JsonPropertyDescription(
            "The model configuration for the models to come up with a reason for each taxonomy category.")
    private ModelProperties discussionModel;
    @JsonProperty("referee-model")
    @JsonPropertyDescription("The model configuration to use for the referee step.")
    private ModelProperties refereeModel;

    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        if (discussionModel.getClient() instanceof LocalClient) {
            throw new UnsupportedOperationException("LocalClient is not supported for DiscussionStrategy");
        }
        if (discussionModel.getResponseFormat() != ResponseFormat.JSON_OBJECT) {
            throw new UnsupportedOperationException(
                    "Only ResponseFormat.JSON_OBJECT is supported for the Discussion Model for the "
                            + "DiscussionStrategy, because the Referee Model "
                            + "needs to have an explanation from the Discussion Model available.");
        }

        String discussionPrompt = PromptUtils.replacePrompt(request, discussionModel.getPrompt(),
                                                            discussionModel.getResponseFormat(),
                                                            discussionModel.isInjectResponseFormat(),
                                                            request.getTaxonomy(),
                                                            discussionModel.getRaqProperties(),
                                                            ClassificationResult.class
        );
        List<ClassificationResult> discussions = new ArrayList<>();
        request.getTaxonomy().getCategories().parallelStream().forEach(category -> {
            String prompt = discussionPrompt.replace(TARGET_LABEL_PLACEHOLDER, category.getName());
            ClassificationResult result = discussionModel.getClient().makeRequest(prompt, ClassificationResult.class);
            if (result != null) {
                result.setName(category.getName());
                discussions.add(result);
            }
        });

        String refereePrompt = PromptUtils.replacePrompt(request,
                                                         refereeModel.getPrompt(),
                                                         refereeModel.getResponseFormat(),
                                                         refereeModel.isInjectResponseFormat(),
                                                         request.getTaxonomy(),
                                                         refereeModel.getRaqProperties(),
                                                         ClassificationResult.class
        ).replace(REASONS_PLACEHOLDER, buildReasonsForEachType(discussions));

        return refereeModel.getClient()
                           .makeRequest(refereePrompt, ClassificationResult.class);
    }

    private String buildReasonsForEachType(List<ClassificationResult> discussions) {
        StringBuilder sb = new StringBuilder();
        for (ClassificationResult discussion : discussions) {
            sb.append("Reasons for *").append(discussion.getName()).append("* are: ");
            sb.append(discussion.getExplanation()).append("\n");
        }
        return sb.toString();
    }

}
