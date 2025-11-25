package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.model.config.ResponseFormat;
import de.tum.claritypipeline.utils.PromptUtils;
import de.tum.clarityutils.SerializationUtils;
import lombok.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Strategy that performs multi-model classification with configurable decision aggregation.
 * <p>
 * Executes classification across multiple models and aggregates results using:
 * - MAJORITY_VOTE: selects the most frequently predicted label
 * - CONFIDENCE_WEIGHTED: selects based on summed confidence scores
 * - PRIORITY_ORDER: selects based on a predefined priority list
 * <p>
 * Supports both structured JSON responses and plain text responses.
 */
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class MultiStrategy implements ClassificationStrategy {
    @JsonProperty("models")
    @JsonPropertyDescription("The model configurations to use for classification.")
    private List<ModelProperties> models;

    @JsonProperty("decision-type")
    @JsonPropertyDescription("The decision type to aggregate model results.")
    private DecisionType decisionType = DecisionType.MAJORITY_VOTE;

    @JsonProperty("fallback-decision-type")
    @JsonPropertyDescription("The fallback decision type to use if the primary decision type cannot produce a result.")
    private DecisionType fallbackDecisionType = DecisionType.PRIORITY_ORDER;

    @JsonProperty("priority-order")
    @JsonPropertyDescription("The priority order of labels for PRIORITY_ORDER decision type.")
    private List<String> priorityOrder = new ArrayList<>();

    /**
     * Execute the multi-model classification with decision aggregation.
     *
     * @param request the classification request containing text and taxonomy.
     * @return a ClassificationResult representing the aggregated prediction.
     */
    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        List<ClassificationResult> results = collectModelResults(request);

        // Apply primary decision type
        List<ClassificationResult> filteredResults = applyDecisionType(results, decisionType);
        if (hasUniqueResult(filteredResults)) {
            return filteredResults.getFirst();
        }

        // Apply fallback decision type
        filteredResults = applyDecisionType(filteredResults, fallbackDecisionType);
        return filteredResults.stream()
                              .findFirst()
                              .orElseThrow(() -> new IllegalStateException(
                                      "No valid decision could be made from the model results."));
    }

    /**
     * Collects classification results from all configured models.
     */
    private List<ClassificationResult> collectModelResults(ClassificationRequest request) {
        return models.stream()
                     .map(model -> executeModelRequest(request, model))
                     .collect(Collectors.toList());
    }

    /**
     * Executes a single model request and returns the classification result.
     */
    private ClassificationResult executeModelRequest(ClassificationRequest request, ModelProperties model) {
        String prompt = buildPrompt(request, model);

        if (model.getResponseFormat() == ResponseFormat.JSON_OBJECT) {
            return model.getClient().makeRequest(prompt, ClassificationResult.class);
        } else {
            String response = model.getClient().makeRequest(prompt);
            return ClassificationResult.builder().name(response).build();
        }
    }

    /**
     * Builds the prompt for a specific model.
     */
    private String buildPrompt(ClassificationRequest request, ModelProperties model) {
        return switch (model.getClient()) {
            case LocalClient ignore -> SerializationUtils.serialize(request);
            default -> PromptUtils.replacePrompt(
                    request,
                    model.getPrompt(),
                    model.getResponseFormat(),
                    model.isInjectResponseFormat(),
                    request.getTaxonomy(),
                    model.getRaqProperties(),
                    ClassificationResult.class
            );
        };
    }

    /**
     * Applies the specified decision type to the results.
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
     * Checks if all results have the same label.
     */
    private boolean hasUniqueResult(List<ClassificationResult> results) {
        return results.stream()
                      .map(ClassificationResult::getName)
                      .distinct()
                      .count() == 1;
    }

    /**
     * Selects results based on majority vote.
     * Returns all results with the most common label(s).
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
     * Selects results based on summed confidence scores.
     * Returns all results with the highest confidence sum(s).
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
     * Generic method to filter results by top score.
     * Returns all results whose label has the maximum score.
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
     * Selects result based on predefined priority order.
     * Returns the first result matching the priority list, or the first available result.
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
     * Decision types for aggregating multi-model classification results.
     */
    public enum DecisionType {
        /**
         * Select based on most frequent prediction
         */
        MAJORITY_VOTE,
        /**
         * Select based on highest summed confidence scores
         */
        CONFIDENCE_WEIGHTED,
        /**
         * Select based on predefined priority order
         */
        PRIORITY_ORDER
    }
}