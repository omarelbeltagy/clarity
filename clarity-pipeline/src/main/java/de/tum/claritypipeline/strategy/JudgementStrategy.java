package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.classification.JudgementResult;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.model.config.ResponseFormat;
import de.tum.claritypipeline.utils.PromptUtils;
import lombok.*;

/**
 * Strategy that performs a two-step classification: an initial classification
 * followed by a judgement step which can confirm or override the initial result.
 * <p>
 * The workflow:
 * 1) Use classificationModel to produce an initial ClassificationResult.
 * 2) Use judgementModel to evaluate (and possibly override) the initial result.
 * <p>
 * This strategy requires that the classificationModel returns structured JSON
 * output (ResponseFormat.JSON_OBJECT) because the judgement model receives the
 * classification explanation as part of its prompt.
 */
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class JudgementStrategy implements ClassificationStrategy {
    /**
     * Model configuration used for the first classification step.
     *
     * This model must produce a JSON_OBJECT response that can be deserialized
     * into a ClassificationResult so that the judgement model can see the
     * explanation and other structured fields.
     */
    @JsonProperty("classification-model")
    @JsonPropertyDescription("The model configuration to use for the initial classification.")
    private ModelProperties classificationModel;

    /**
     * Model configuration used for the judgement step.
     *
     * The judgement model receives the initial classification result (including
     * the explanation) and returns a JudgementResult indicating whether it
     * confirms the initial classification or supplies an alternative.
     */
    @JsonProperty("judgement-model")
    @JsonPropertyDescription("The model configuration to use for the judgement step.")
    private ModelProperties judgementModel;

    /**
     * Execute the two-step judgement strategy.
     *
     * The method performs validation of the provided model configurations and
     * then sequentially calls the configured clients to obtain the initial
     * ClassificationResult and the subsequent JudgementResult. Based on the
     * judgement it returns either a merged ClassificationResult containing the
     * original classification (with added judgement metadata) or the judgement's
     * chosen class.
     *
     * Important behaviors and exceptions:
     * - If the classificationModel uses a LocalClient, an UnsupportedOperationException
     *   is thrown because local models are not supported for this strategy.
     * - If the classificationModel.responseFormat is not JSON_OBJECT, an
     *   UnsupportedOperationException is thrown because the judgement step
     *   requires structured explanation content.
     *
     * @param request the classification request containing input text and taxonomy;
     *                must not be null.
     * @return a ClassificationResult representing the final chosen label and
     *         associated metadata (confidence, explanations, judgement fields).
     * @throws UnsupportedOperationException if configuration is incompatible
     *                                       with the judgement workflow.
     */
    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        if (classificationModel.getClient() instanceof LocalClient) {
            throw new UnsupportedOperationException("LocalClient is not supported for JudgementStrategy");
        }
        if (classificationModel.getResponseFormat() != ResponseFormat.JSON_OBJECT) {
            throw new UnsupportedOperationException(
                    "Only ResponseFormat.JSON_OBJECT is supported for the Classification Model for the "
                            + "JudgementStrategy, because the Judgement Model "
                            + "needs to have an explanation from the Classification Model available.");
        }
        String prompt = PromptUtils.replacePrompt(request, classificationModel.getPrompt(),
                                                  classificationModel.getResponseFormat(),
                                                  classificationModel.isInjectResponseFormat(),
                                                  request.getTaxonomy(),
                                                  classificationModel.getRaqProperties(),
                                                  ClassificationResult.class
        );
        ClassificationResult initialResult = classificationModel.getClient()
                                                                .makeRequest(prompt, ClassificationResult.class);

        String judgementPrompt = PromptUtils.replaceJudgementPrompt(request,
                                                                    initialResult,
                                                                    judgementModel.getPrompt(),
                                                                    judgementModel.getResponseFormat(),
                                                                    judgementModel.isInjectResponseFormat(),
                                                                    request.getTaxonomy(),
                                                                    judgementModel.getRaqProperties(),
                                                                    JudgementResult.class
        );

        JudgementResult judgementResult = judgementModel.getClient()
                                                        .makeRequest(judgementPrompt, JudgementResult.class);

        if (judgementResult.isConfirmed() || judgementResult.getName().equals(initialResult.getName())) {
            return ClassificationResult.builder()
                                       .name(initialResult.getName())
                                       .explanation(initialResult.getExplanation())
                                       .confidence(initialResult.getConfidence())
                                       .judgementConfidence(judgementResult.getConfidence())
                                       .judgementExplanation(initialResult.getJudgementExplanation())
                                       .build();
        } else {
            return ClassificationResult.builder()
                                       .name(judgementResult.getName())
                                       .explanation(judgementResult.getExplanation())
                                       .confidence(judgementResult.getConfidence())
                                       .build();
        }
    }
}
