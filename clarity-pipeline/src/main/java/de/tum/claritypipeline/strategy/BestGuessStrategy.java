package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonIgnore;
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
@Node(label = "BestGuessStrategy")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class BestGuessStrategy extends Neo4jNode implements ClassificationStrategy {
    @JsonProperty("model")
    @JsonPropertyDescription("The model configuration to use for classification.")
    @Neo4jIgnore
    private ModelProperties model;

    @JsonIgnore
    @Neo4jIgnore
    private Taxonomy taxonomy;

    @JsonProperty("k")
    @JsonPropertyDescription("The number of guesses for the model")
    private int k = 3;

    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        validateConfiguration();
        String prompt = PromptUtils.replacePrompt(request, model, BestGuessClassificationResult.JSON_SCHEME)
                                   .replace("{k}", String.valueOf(k));
        BestGuessClassificationResult result = model.getClient()
                                                    .makeRequest(prompt, BestGuessClassificationResult.class);
        if (result == null || result.getTopLabels() == null || result.getTopLabels().isEmpty()) {
            throw new IllegalStateException();
        }
        List<String> labels = result.getTopLabels().stream()
                                    .map(label -> {
                                        Taxonomy.Category category = taxonomy.getCategories().stream()
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
     * Validates that the model configuration is compatible with best guess strategy.
     */
    private void validateConfiguration() {
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

        if (taxonomy == null) {
            throw new UnsupportedOperationException(
                    "Taxonomy is not set for BestGuessStrategy"
            );
        }

        if (taxonomy.getMapping() == null || !taxonomy.getMapping().isEnabled()) {
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
