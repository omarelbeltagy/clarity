package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.classification.JudgementResult;
import de.tum.claritypipeline.model.config.ModelProperties;
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
@Node(label = "JudgementStrategy")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class JudgementStrategy extends Neo4jNode implements ClassificationStrategy {
    @JsonIgnore
    private static final String PLACEHOLDER_CLASSIFICATION_RESULT = "{classification_result}";

    @JsonProperty("classification-model")
    @JsonPropertyDescription("The model configuration to use for the initial classification.")
    private ModelProperties classificationModel;

    @JsonProperty("judgement-model")
    @JsonPropertyDescription("The model configuration to use for the judgement step.")
    private ModelProperties judgementModel;

    /**
     * Execute the two-step judgement strategy.
     *
     * @param request the classification request containing input text and taxonomy.
     * @return a ClassificationResult representing the final chosen label and metadata.
     * @throws UnsupportedOperationException if configuration is incompatible.
     */
    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        validateConfiguration();

        ClassificationResult initialResult = performInitialClassification(request);
        JudgementResult judgementResult = performJudgement(request, initialResult);

        return mergeResults(initialResult, judgementResult);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T extends Neo4jNode> T getClassificationStrategyNode() {
        return (T) this;
    }

    /**
     * Validates that the model configuration is compatible with judgement strategy.
     */
    private void validateConfiguration() {
        if (classificationModel.getClient() instanceof LocalClient) {
            throw new UnsupportedOperationException(
                    "LocalClient is not supported for JudgementStrategy"
            );
        }

        if (classificationModel.getResponseFormat() != ModelProperties.ResponseFormat.JSON_OBJECT) {
            throw new UnsupportedOperationException(
                    "Only ResponseFormat.JSON_OBJECT is supported for the Classification Model in " +
                            "JudgementStrategy, because the Judgement Model needs access to the explanation."
            );
        }
    }

    /**
     * Performs the initial classification step.
     */
    private ClassificationResult performInitialClassification(ClassificationRequest request) {
        String prompt = PromptUtils.replacePrompt(
                request,
                classificationModel,
                ClassificationResult.JSON_SCHEME
        );

        return classificationModel.getClient()
                                  .makeRequest(prompt, ClassificationResult.class);
    }

    /**
     * Performs the judgement step to evaluate the initial classification.
     */
    private JudgementResult performJudgement(
            ClassificationRequest request,
            ClassificationResult initialResult
    ) {

        String judgementPrompt = buildJudgementPrompt(request, initialResult);

        return judgementModel.getClient()
                             .makeRequest(judgementPrompt, JudgementResult.class);
    }

    /**
     * Builds the prompt for the judgement model, including the initial classification result.
     */
    private String buildJudgementPrompt(
            ClassificationRequest request,
            ClassificationResult initialResult
    ) {

        String prompt = PromptUtils.replacePrompt(
                request,
                judgementModel,
                JudgementResult.JSON_SCHEME
        );

        String classificationResultStr = formatClassificationResult(initialResult);
        return prompt.replace(PLACEHOLDER_CLASSIFICATION_RESULT, classificationResultStr);
    }

    /**
     * Formats the classification result for inclusion in the judgement prompt.
     */
    private String formatClassificationResult(ClassificationResult result) {
        StringBuilder sb = new StringBuilder();
        sb.append("Name: ").append(result.getName()).append("\n");

        if (result.getExplanation() != null && !result.getExplanation().isEmpty()) {
            sb.append("Explanation: ").append(result.getExplanation()).append("\n");
        }

        return sb.toString();
    }

    /**
     * Merges initial classification and judgement results into final output.
     * Returns the confirmed initial result with judgement metadata, or the overridden result.
     */
    private ClassificationResult mergeResults(
            ClassificationResult initialResult,
            JudgementResult judgementResult
    ) {

        if (isConfirmed(initialResult, judgementResult)) {
            return buildConfirmedResult(initialResult, judgementResult);
        } else {
            return buildOverriddenResult(judgementResult);
        }
    }

    /**
     * Checks if the judgement confirms the initial classification.
     */
    private boolean isConfirmed(ClassificationResult initial, JudgementResult judgement) {
        return judgement.isConfirmed() || judgement.getName().equals(initial.getName());
    }

    /**
     * Builds result when judgement confirms initial classification.
     */
    private ClassificationResult buildConfirmedResult(
            ClassificationResult initial,
            JudgementResult judgement
    ) {

        return ClassificationResult.builder()
                                   .name(initial.getName())
                                   .explanation(initial.getExplanation())
                                   .confidence(initial.getConfidence())
                                   .judgementConfidence(judgement.getConfidence())
                                   .judgementExplanation(judgement.getExplanation())
                                   .build();
    }

    /**
     * Builds result when judgement overrides initial classification.
     */
    private ClassificationResult buildOverriddenResult(JudgementResult judgement) {
        return ClassificationResult.builder()
                                   .name(judgement.getName())
                                   .explanation(judgement.getExplanation())
                                   .confidence(judgement.getConfidence())
                                   .build();
    }
}