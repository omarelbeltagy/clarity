package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.classification.JudgementResult;
import de.tum.claritypipeline.model.config.GlobalConfig;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.model.relation.HasClassificationModel;
import de.tum.claritypipeline.model.relation.HasJudgementModel;
import de.tum.claritypipeline.utils.PromptUtils;
import de.tum.clarityutils.AfterDeserialization;
import lombok.*;

import java.util.Map;

/**
 * Two-phase classification strategy where an initial classification is reviewed by a judgement model.
 * <p>
 * This strategy implements a classify-then-verify approach that can improve classification accuracy
 * by having a second model review and potentially override the initial prediction:
 * <ol>
 *   <li><b>Classification Phase</b>: Initial model classifies the input and provides reasoning</li>
 *   <li><b>Judgement Phase</b>: Judgement model reviews the classification with its reasoning
 *       and either confirms or overrides the decision</li>
 * </ol>
 *
 * <h2>Classification Process</h2>
 * <pre>
 * Input: Question & Answer
 *    ↓
 * Classification Model
 *    → Prediction: Category A
 *    → Explanation: "Because of X, Y, Z..."
 *    → Confidence: 0.85
 *    ↓
 * Judgement Model (receives prediction + explanation)
 *    → Reviews reasoning
 *    → Decision: CONFIRMED or OVERRIDE
 *    → If override: New Category + Explanation
 *    ↓
 * Output: Final Classification
 * </pre>
 *
 * <h2>Result Merging</h2>
 * <ul>
 *   <li><b>If Confirmed</b>: Returns initial classification with added judgement metadata
 *       (judgementConfidence, judgementExplanation)</li>
 *   <li><b>If Overridden</b>: Returns new classification from judgement model</li>
 * </ul>
 *
 * <h2>Use Cases</h2>
 * <ul>
 *   <li>High-stakes classifications requiring verification</li>
 *   <li>Combining strengths of different models (e.g., fast classifier + careful judge)</li>
 *   <li>Scenarios where explicit reasoning validation is important</li>
 * </ul>
 *
 * <h2>Requirements</h2>
 * <ul>
 *   <li>Classification model must use JSON_OBJECT format (judgement needs explanation)</li>
 *   <li>LocalClient is not supported (requires remote model API)</li>
 *   <li>Judgement prompt must include {classification_result} placeholder</li>
 * </ul>
 *
 * @see ClassificationResult
 * @see JudgementResult
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

    /**
     * Model configuration for performing the initial classification.
     * This model should provide both a predicted label and an explanation
     * for the judgement model to review.
     */
    @JsonProperty("classification-model")
    @JsonPropertyDescription("The model configuration to use for the initial classification.")
    @Neo4jIgnore
    private ModelProperties classificationModel;

    /**
     * Model configuration for reviewing and potentially overriding the initial classification.
     * This model receives the initial prediction with its explanation and makes
     * a final decision about the correct classification.
     */
    @JsonProperty("judgement-model")
    @JsonPropertyDescription("The model configuration to use for the judgement step.")
    @Neo4jIgnore
    private ModelProperties judgementModel;

    /**
     * Executes the two-phase judgement-based classification strategy.
     * <p>
     * Process:
     * <ol>
     *   <li>Validates configuration (JSON format, not LocalClient)</li>
     *   <li><b>Classification Phase</b>:
     *     <ul>
     *       <li>Sends input to classification model</li>
     *       <li>Receives prediction with explanation and confidence</li>
     *     </ul>
     *   </li>
     *   <li><b>Judgement Phase</b>:
     *     <ul>
     *       <li>Formats initial result for judgement prompt</li>
     *       <li>Sends to judgement model with classification result</li>
     *       <li>Receives judgement decision (confirm/override)</li>
     *     </ul>
     *   </li>
     *   <li><b>Merge Results</b>:
     *     <ul>
     *       <li>If confirmed: Enhances initial result with judgement metadata</li>
     *       <li>If overridden: Returns new classification from judgement</li>
     *     </ul>
     *   </li>
     * </ol>
     *
     * @param request the classification request with question, answer, and taxonomy
     * @return merged classification result representing the final decision
     * @throws UnsupportedOperationException if configuration requirements are not met
     */
    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        validateConfiguration();

        ClassificationResult initialResult = performInitialClassification(request);
        JudgementResult judgementResult = performJudgement(request, initialResult);

        return mergeResults(initialResult, judgementResult);
    }

    @AfterDeserialization
    private void createNode() {
        String query = """
                MATCH(jm:%s)<-[:%s]-(n:%s)-[:%s]->(cm:%s)
                WHERE elementId(cm) = $classificationModelPropertiesId
                AND elementId(jm) = $judgementModelPropertiesId
                RETURN n
                """.formatted(
                Neo4jNode.getLabel(ModelProperties.class),
                Neo4jRelation.getType(HasJudgementModel.class),
                Neo4jNode.getLabel(JudgementStrategy.class),
                Neo4jRelation.getType(HasClassificationModel.class),
                Neo4jNode.getLabel(ModelProperties.class));

        JudgementStrategy existingNode = GlobalConfig.NEO4J_CLIENT.executeQuery(query,
                                                                                Map.of("classificationModelPropertiesId",
                                                                                       classificationModel.getElementId(),
                                                                                       "judgementModelPropertiesId",
                                                                                       judgementModel.getElementId()),
                                                                                JudgementStrategy.class).stream()
                                                                  .findFirst()
                                                                  .orElse(null);

        if (existingNode != null && allRelationsExist(existingNode)) {
            setElementId(existingNode.getElementId());
            return;
        }

        GlobalConfig.NEO4J_CLIENT.saveNode(this);
        createRelationIfNeeded(classificationModel, HasClassificationModel.builder().build());
        createRelationIfNeeded(judgementModel, HasJudgementModel.builder().build());
    }

    private boolean allRelationsExist(JudgementStrategy existingNode) {
        boolean classificationModelOk =
                GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(), classificationModel.getElementId(),
                                                       HasClassificationModel.class)
                        != null;

        boolean judgementModelOk =
                GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(), judgementModel.getElementId(),
                                                       HasJudgementModel.class)
                        != null;

        return classificationModelOk && judgementModelOk;
    }

    private <T extends Neo4jRelation, N extends Neo4jNode> void createRelationIfNeeded(
            N targetNode, T relation) {
        if (targetNode == null) return;
        relation.setStartNodeId(this.getElementId());
        relation.setEndNodeId(targetNode.getElementId());
        GlobalConfig.NEO4J_CLIENT.createRelation(relation);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T extends Neo4jNode> T getClassificationStrategyNode() {
        return (T) this;
    }

    /**
     * Validates that the configuration supports judgement strategy requirements.
     * <p>
     * Ensures:
     * <ul>
     *   <li>Classification model is not LocalClient</li>
     *   <li>Classification model uses JSON_OBJECT format (required for explanation access)</li>
     * </ul>
     *
     * @throws UnsupportedOperationException if any requirement is not met
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
     * Performs the initial classification step using the classification model.
     *
     * @param request the classification request
     * @return initial classification result with label, explanation, and confidence
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
     *
     * @param request the original classification request
     * @param initialResult the initial classification to be judged
     * @return judgement result indicating confirmation or override
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
     * <p>
     * Replaces the {classification_result} placeholder with formatted information
     * about the initial prediction and its reasoning.
     *
     * @param request the classification request
     * @param initialResult the initial classification result
     * @return complete prompt for the judgement model
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
     * Formats the initial classification result for inclusion in the judgement prompt.
     * <p>
     * Format includes label name and explanation (if available).
     *
     * @param result the initial classification result
     * @return formatted string representation
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
     * Merges initial classification and judgement results into the final output.
     * <p>
     * Logic:
     * <ul>
     *   <li>If judgement confirms initial result: Returns initial result enhanced
     *       with judgement confidence and explanation</li>
     *   <li>If judgement overrides: Returns new classification from judgement result</li>
     * </ul>
     *
     * @param initialResult the initial classification
     * @param judgementResult the judgement decision
     * @return merged final classification result
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
     * <p>
     * Confirmation is determined by:
     * <ul>
     *   <li>Judgement explicitly marked as confirmed, OR</li>
     *   <li>Judgement result name matches initial result name</li>
     * </ul>
     *
     * @param initial the initial classification
     * @param judgement the judgement decision
     * @return true if judgement confirms initial classification
     */
    private boolean isConfirmed(ClassificationResult initial, JudgementResult judgement) {
        return judgement.isConfirmed() || judgement.getName().equals(initial.getName());
    }

    /**
     * Builds the final result when judgement confirms initial classification.
     * <p>
     * Preserves initial prediction and adds judgement metadata for traceability.
     *
     * @param initial the confirmed initial classification
     * @param judgement the confirming judgement
     * @return classification result with judgement metadata
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
     * Builds the final result when judgement overrides initial classification.
     * <p>
     * Uses the new label, explanation, and confidence from the judgement.
     *
     * @param judgement the overriding judgement
     * @return new classification result from judgement
     */
    private ClassificationResult buildOverriddenResult(JudgementResult judgement) {
        return ClassificationResult.builder()
                                   .name(judgement.getName())
                                   .explanation(judgement.getExplanation())
                                   .confidence(judgement.getConfidence())
                                   .build();
    }
}