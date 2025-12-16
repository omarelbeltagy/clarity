package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.utils.PromptUtils;
import de.tum.clarityutils.SerializationUtils;
import lombok.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Multi-model classification strategy with configurable decision aggregation mechanisms.
 * <p>
 * This strategy queries multiple classification models and aggregates their predictions
 * using various decision types to produce a final classification. It supports ensemble
 * learning approaches that can improve robustness and accuracy over single-model predictions.
 *
 * <h2>Classification Process</h2>
 * <pre>
 * Input: Question & Answer
 *    ↓
 * Model 1 → Prediction A (confidence: 0.8)
 * Model 2 → Prediction B (confidence: 0.6)
 * Model 3 → Prediction A (confidence: 0.9)
 *    ↓
 * Decision Aggregation (based on DecisionType)
 *    ↓
 * Output: Final Classification
 * </pre>
 *
 * <h2>Decision Types</h2>
 * <ul>
 *   <li><b>MAJORITY_VOTE</b>: Selects the most frequently predicted label
 *       <ul>
 *         <li>Simple democratic voting across models</li>
 *         <li>Effective when models are equally reliable</li>
 *         <li>Handles ties by applying fallback decision type</li>
 *       </ul>
 *   </li>
 *   <li><b>CONFIDENCE_WEIGHTED</b>: Selects label with highest summed confidence scores
 *       <ul>
 *         <li>Weights predictions by model confidence</li>
 *         <li>Useful when models provide calibrated confidence scores</li>
 *         <li>Handles ties by applying fallback decision type</li>
 *       </ul>
 *   </li>
 *   <li><b>PRIORITY_ORDER</b>: Selects based on predefined priority list
 *       <ul>
 *         <li>Uses configured priority order to break ties</li>
 *         <li>Allows encoding domain knowledge about label importance</li>
 *         <li>Falls back to first available prediction if no match</li>
 *       </ul>
 *   </li>
 * </ul>
 *
 * <h2>Fallback Mechanism</h2>
 * If the primary decision type results in a tie (multiple labels with equal scores),
 * the strategy automatically applies the fallback decision type to resolve the ambiguity.
 *
 * <h2>Use Cases</h2>
 * <ul>
 *   <li>Ensemble learning to improve classification accuracy</li>
 *   <li>Combining models with different strengths (e.g., speed vs. accuracy)</li>
 *   <li>Robust classification by reducing single-model biases</li>
 *   <li>A/B testing multiple model configurations</li>
 * </ul>
 *
 * <h2>Performance Considerations</h2>
 * This strategy makes N API calls (where N = number of models). Consider costs
 * and latency when configuring multiple models.
 *
 * @see DecisionType
 * @see ClassificationResult
 */
@Node(label = "MultiStrategy")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class MultiStrategy extends Neo4jNode implements ClassificationStrategy {
    /**
     * List of model configurations to query for classification.
     * Each model is queried independently and results are aggregated.
     */
    @JsonProperty("models")
    @JsonPropertyDescription("The model configurations to use for classification.")
    private List<ModelProperties> models;

    /**
     * Primary decision type for aggregating model predictions.
     * Applied first to determine the final classification.
     */
    @JsonProperty("decision-type")
    @JsonPropertyDescription("The decision type to aggregate model results.")
    private DecisionType decisionType = DecisionType.MAJORITY_VOTE;

    /**
     * Fallback decision type used when primary decision results in a tie.
     * Ensures a deterministic final decision even in ambiguous cases.
     */
    @JsonProperty("fallback-decision-type")
    @JsonPropertyDescription("The fallback decision type to use if the primary decision type cannot produce a result.")
    private DecisionType fallbackDecisionType = DecisionType.PRIORITY_ORDER;

    /**
     * Priority order of labels for PRIORITY_ORDER decision type.
     * Labels appearing earlier in the list are preferred in case of ties.
     */
    @JsonProperty("priority-order")
    @JsonPropertyDescription("The priority order of labels for PRIORITY_ORDER decision type.")
    private List<String> priorityOrder = new ArrayList<>();

    /**
     * Executes multi-model classification with decision aggregation.
     * <p>
     * Workflow:
     * <ol>
     *   <li><b>Collect Results</b>: Query all configured models in parallel</li>
     *   <li><b>Apply Primary Decision</b>: Use primary decision type to filter/rank results</li>
     *   <li><b>Check Uniqueness</b>: If single winner, return immediately</li>
     *   <li><b>Apply Fallback</b>: If tie, use fallback decision type</li>
     *   <li><b>Return Final Result</b>: Return first result from filtered set</li>
     * </ol>
     *
     * @param request the classification request with question, answer, and taxonomy
     * @return aggregated classification result
     * @throws IllegalStateException if no valid decision can be made after fallback
     */
    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        List<ClassificationResult> results = collectModelResults(request);

        // Apply primary decision type
        List<ClassificationResult> filteredResults = applyDecisionType(results, decisionType);
        if (hasUniqueResult(filteredResults)) {
            return filteredResults.get(0);
        }

        // Apply fallback decision type
        filteredResults = applyDecisionType(filteredResults, fallbackDecisionType);
        return filteredResults.stream()
                              .findFirst()
                              .orElseThrow(() -> new IllegalStateException(
                                      "No valid decision could be made from the model results."));
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T extends Neo4jNode> T getClassificationStrategyNode() {
        return (T) this;
    }

    /**
     * Collects classification results from all configured models.
     * <p>
     * Executes requests to all models and collects their predictions.
     *
     * @param request the classification request
     * @return list of classification results from all models
     */
    private List<ClassificationResult> collectModelResults(ClassificationRequest request) {
        return models.stream()
                     .map(model -> executeModelRequest(request, model))
                     .collect(Collectors.toList());
    }

    /**
     * Executes a classification request for a single model.
     * <p>
     * Handles both JSON and plain text response formats:
     * <ul>
     *   <li>JSON_OBJECT: Deserializes to ClassificationResult</li>
     *   <li>Plain text: Creates ClassificationResult with label only</li>
     * </ul>
     *
     * @param request the classification request
     * @param model the model configuration to use
     * @return classification result from the model
     */
    private ClassificationResult executeModelRequest(ClassificationRequest request, ModelProperties model) {
        String prompt = buildPrompt(request, model);

        if (model.getResponseFormat() == ModelProperties.ResponseFormat.JSON_OBJECT) {
            return model.getClient().makeRequest(prompt, ClassificationResult.class);
        } else {
            String response = model.getClient().makeRequest(prompt);
            return ClassificationResult.builder().name(response).build();
        }
    }

    /**
     * Builds the appropriate prompt for a specific model.
     * <p>
     * For LocalClient: Serializes the entire request object.
     * For remote clients: Uses PromptUtils to build formatted prompt.
     *
     * @param request the classification request
     * @param model the model configuration
     * @return formatted prompt string
     */
    private String buildPrompt(ClassificationRequest request, ModelProperties model) {
        return switch (model.getClient()) {
            case LocalClient ignore -> SerializationUtils.serialize(request);
            default -> PromptUtils.replacePrompt(
                    request,
                    model,
                    ClassificationResult.JSON_SCHEME
            );
        };
    }

    /**
     * Applies the specified decision type to aggregate and filter results.
     * <p>
     * Returns a filtered list containing only the top-scoring result(s) according
     * to the decision type logic. May return multiple results if there's a tie.
     *
     * @param results list of classification results from models
     * @param type the decision type to apply
     * @return filtered list of top result(s)
     */
    private List<ClassificationResult> applyDecisionType(
            List<ClassificationResult> results,
            DecisionType type
    ) {

        return switch (type) {
            case MAJORITY_VOTE -> majorityVote(results);
            case CONFIDENCE_WEIGHTED -> confidenceVote(results);
            case PRIORITY_ORDER -> List.of(priorityOrder(results));
        };
    }

    /**
     * Checks if all filtered results have the same label (no tie).
     *
     * @param results filtered results
     * @return true if all results have identical labels
     */
    private boolean hasUniqueResult(List<ClassificationResult> results) {
        return results.stream()
                      .map(ClassificationResult::getName)
                      .distinct()
                      .count() == 1;
    }

    /**
     * Implements majority vote decision: selects label(s) with highest frequency.
     * <p>
     * Counts how many models predicted each label and returns all results
     * matching the most frequent label(s).
     *
     * @param results list of classification results
     * @return results with most frequently predicted label(s)
     */
    private List<ClassificationResult> majorityVote(List<ClassificationResult> results) {
        Map<String, Long> counts = results.stream()
                                          .collect(Collectors.groupingBy(
                                                  ClassificationResult::getName,
                                                  Collectors.counting()
                                          ));

        long maxCount = counts.values().stream()
                              .max(Long::compare)
                              .orElse(0L);

        return filterResultsByTopScore(results, counts, maxCount);
    }

    /**
     * Implements confidence-weighted decision: selects label(s) with highest summed confidence.
     * <p>
     * Sums confidence scores per label across all models and returns results
     * matching the label(s) with highest total confidence.
     *
     * @param results list of classification results
     * @return results with highest confidence sum(s)
     */
    private List<ClassificationResult> confidenceVote(List<ClassificationResult> results) {
        Map<String, Double> confidenceSums = results.stream()
                                                    .collect(Collectors.groupingBy(
                                                            ClassificationResult::getName,
                                                            Collectors.summingDouble(
                                                                    r -> r.getConfidence() != null ? r.getConfidence()
                                                                            : 0.0)
                                                    ));

        double maxConfidence = confidenceSums.values().stream()
                                             .max(Double::compare)
                                             .orElse(0.0);

        return filterResultsByTopScore(results, confidenceSums, maxConfidence);
    }

    /**
     * Generic method to filter results based on top score(s).
     * <p>
     * Returns all results whose label achieved the maximum score.
     * Used by both majorityVote and confidenceVote.
     *
     * @param results list of classification results
     * @param scores map of label to score
     * @param maxScore the maximum score value
     * @param <T> numeric type of the score
     * @return results matching the top score
     */
    private <T extends Number> List<ClassificationResult> filterResultsByTopScore(
            List<ClassificationResult> results,
            Map<String, T> scores,
            T maxScore
    ) {

        List<String> topLabels = scores.entrySet().stream()
                                       .filter(entry -> entry.getValue().equals(maxScore))
                                       .map(Map.Entry::getKey)
                                       .toList();

        return results.stream()
                      .filter(result -> topLabels.contains(result.getName()))
                      .collect(Collectors.toList());
    }

    /**
     * Implements priority order decision: selects first result matching priority list.
     * <p>
     * Iterates through the configured priority order and returns the first result
     * whose label matches. Falls back to first available result if no match found.
     *
     * @param results list of classification results
     * @return result matching highest priority, or first available result
     */
    private ClassificationResult priorityOrder(List<ClassificationResult> results) {
        return priorityOrder.stream()
                            .flatMap(priority -> results.stream()
                                                        .filter(result -> result.getName().equals(priority)))
                            .findFirst()
                            .or(() -> results.stream().findFirst())
                            .orElse(null);
    }

    /**
     * Enumeration of decision types for aggregating multi-model classification results.
     */
    public enum DecisionType {
        /**
         * Select label based on most frequent prediction across models.
         * Treats all models equally regardless of confidence.
         */
        MAJORITY_VOTE,

        /**
         * Select label based on highest summed confidence scores.
         * Weights predictions by model-reported confidence values.
         */
        CONFIDENCE_WEIGHTED,

        /**
         * Select label based on predefined priority order.
         * Uses configured priority list to resolve ambiguity.
         */
        PRIORITY_ORDER
    }
}