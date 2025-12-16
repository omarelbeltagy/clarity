package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.classification.BestGuessClassificationResult;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.GlobalConfig;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.model.core.Taxonomy;
import de.tum.claritypipeline.model.relation.HasClassificationModel;
import de.tum.claritypipeline.utils.PromptUtils;
import de.tum.clarityutils.AfterDeserialization;
import lombok.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Classification strategy that generates multiple top-k predictions and selects the most common mapped label.
 * <p>
 * This strategy is designed for scenarios where:
 * <ul>
 *   <li>The taxonomy has a mapping configuration enabled</li>
 *   <li>Multiple category predictions can be aggregated to improve accuracy</li>
 *   <li>The model can return structured JSON with top-k predictions</li>
 * </ul>
 *
 * <h2>Classification Process</h2>
 * <ol>
 *   <li><b>Generate Top-K Predictions</b>: Request k best guesses from the model</li>
 *   <li><b>Map to Target Labels</b>: Map each predicted category to its target label via taxonomy mapping</li>
 *   <li><b>Aggregate by Frequency</b>: Count occurrences of each mapped label</li>
 *   <li><b>Select Most Common</b>: Return the most frequently occurring mapped label</li>
 * </ol>
 *
 * <h2>Example</h2>
 * Given k=3 predictions: ["CategoryA", "CategoryB", "CategoryA"] which map to ["LabelX", "LabelY", "LabelX"],
 * the strategy returns "LabelX" as it appears twice.
 *
 * <h2>Requirements</h2>
 * <ul>
 *   <li>Response format must be JSON_OBJECT (to parse structured top-k results)</li>
 *   <li>Taxonomy mapping must be enabled</li>
 *   <li>LocalClient is not supported (requires remote model API)</li>
 * </ul>
 *
 * @see BestGuessClassificationResult
 * @see Taxonomy.Mapping
 */
@Node(label = "BestGuessStrategy")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class BestGuessStrategy extends Neo4jNode implements ClassificationStrategy {
    /**
     * Model configuration for generating top-k predictions.
     * Must support JSON response format to return structured results.
     */
    @JsonProperty("model")
    @JsonPropertyDescription("The model configuration to use for classification.")
    @Neo4jIgnore
    private ModelProperties model;

    /**
     * Number of best guesses to request from the model.
     * Higher values provide more data for aggregation but increase API costs.
     * Typical values: 3-5.
     */
    @JsonProperty("k")
    @JsonPropertyDescription("The number of guesses for the model")
    private int k = 3;

    /**
     * Executes the best-guess classification with label aggregation.
     * <p>
     * Steps:
     * <ol>
     *   <li>Validates configuration (JSON format, mapping enabled, not LocalClient)</li>
     *   <li>Builds prompt with k parameter for top-k predictions</li>
     *   <li>Requests structured response with multiple label predictions</li>
     *   <li>Maps each predicted label to its target via taxonomy mapping</li>
     *   <li>Aggregates mapped labels by frequency</li>
     *   <li>Returns the most common mapped label as final result</li>
     * </ol>
     *
     * @param request the classification request with question, answer, and taxonomy
     * @return classification result with the most frequently mapped label
     * @throws UnsupportedOperationException if configuration requirements are not met
     * @throws IllegalStateException         if no valid labels can be extracted
     */
    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        validateConfiguration(request);
        String prompt = PromptUtils.replacePrompt(request, model, BestGuessClassificationResult.JSON_SCHEME)
                                   .replace("{k}", String.valueOf(k));
        BestGuessClassificationResult result = model.getClient()
                                                    .makeRequest(prompt, BestGuessClassificationResult.class);
        if (result == null || result.getTopLabels() == null || result.getTopLabels().isEmpty()) {
            throw new IllegalStateException();
        }
        List<String> labels = result.getTopLabels().stream()
                                    .map(label -> {
                                        Taxonomy.Category category = request.getTaxonomy().getCategories().stream()
                                                                            .filter(c -> c.getName()
                                                                                          .equalsIgnoreCase(
                                                                                                  label.getName()))
                                                                            .findFirst().orElse(null);
                                        if (category == null) {
                                            return null;
                                        }
                                        if (category.getMapTo() == null) {
                                            return null;
                                        }
                                        return category.getMapTo();
                                    })
                                    .toList();
        if (labels.isEmpty()) {
            throw new IllegalStateException("Failed to extract labels from BestGuessClassificationResult");
        }
        String label = mostCommonLabel(labels);
        return ClassificationResult.builder()
                                   .name(label)
                                   .build();
    }

    /**
     * Determines the most frequently occurring label in the list.
     * <p>
     * In case of ties, returns the first label that achieved the maximum count.
     *
     * @param labels list of mapped labels from top-k predictions
     * @return the most common label
     */
    private String mostCommonLabel(List<String> labels) {
        Map<String, Integer> counts = new HashMap<>();
        for (String label : labels) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        String best = null;
        int bestCount = 0;

        for (String label : labels) {
            int c = counts.get(label);
            if (c > bestCount) {
                best = label;
                bestCount = c;
            }
        }

        return best;
    }


    /**
     * Validates that the configuration supports best-guess strategy requirements.
     * <p>
     * Checks:
     * <ul>
     *   <li>Client is not LocalClient (unsupported)</li>
     *   <li>Response format is JSON_OBJECT (required for structured top-k output)</li>
     *   <li>Taxonomy is provided</li>
     *   <li>Taxonomy mapping is enabled</li>
     * </ul>
     *
     * @param request the classification request
     * @throws UnsupportedOperationException if any requirement is not met
     */
    private void validateConfiguration(ClassificationRequest request) {
        if (model.getClient() instanceof LocalClient) {
            throw new UnsupportedOperationException(
                    "LocalClient is not supported for BestGuessStrategy"
            );
        }

        if (model.getResponseFormat() != ModelProperties.ResponseFormat.JSON_OBJECT) {
            throw new UnsupportedOperationException(
                    "Only ResponseFormat.JSON_OBJECT is supported for the Classification Model in " +
                            "BestGuessStrategy, because the Model needs return multiple labels"
            );
        }

        if (request.getTaxonomy() == null) {
            throw new UnsupportedOperationException(
                    "Taxonomy is not set for BestGuessStrategy"
            );
        }

        if (request.getTaxonomy().getMapping() == null || !request.getTaxonomy().getMapping().isEnabled()) {
            throw new UnsupportedOperationException(
                    "Taxonomy mapping must be enabled for BestGuessStrategy"
            );
        }
    }

    @AfterDeserialization
    private void createNode() {
        String query = """
                MATCH(n:%s)-[:%s]->(m:%s)
                WHERE elementId(m) = $modelPropertiesId
                AND n.k = $k
                RETURN n
                """.formatted(Neo4jNode.getLabel(BestGuessStrategy.class),
                              Neo4jRelation.getType(HasClassificationModel.class),
                              Neo4jNode.getLabel(ModelProperties.class));

        BestGuessStrategy existingNode = GlobalConfig.NEO4J_CLIENT.executeQuery(query, Map.of("modelPropertiesId",
                                                                                              model.getElementId(), "k",
                                                                                              k),
                                                                                BestGuessStrategy.class).stream()
                                                                  .findFirst()
                                                                  .orElse(null);

        if (existingNode != null && allRelationsExist(existingNode)) {
            setElementId(existingNode.getElementId());
            return;
        }

        GlobalConfig.NEO4J_CLIENT.saveNode(this);
        createRelationIfNeeded(model, HasClassificationModel.builder().build());
    }

    private boolean allRelationsExist(BestGuessStrategy existingNode) {
        return GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(), model.getElementId(),
                                                      HasClassificationModel.class)
                != null;
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

}
