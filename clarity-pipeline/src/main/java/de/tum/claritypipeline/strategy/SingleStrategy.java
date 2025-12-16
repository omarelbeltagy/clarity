package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.GlobalConfig;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.model.relation.HasClassificationModel;
import de.tum.claritypipeline.utils.PromptUtils;
import de.tum.clarityutils.AfterDeserialization;
import de.tum.clarityutils.SerializationUtils;
import lombok.*;

import java.util.Map;

/**
 * Simple single-model classification strategy that queries one model for direct prediction.
 * <p>
 * This is the most straightforward classification approach, making a single API call to
 * a configured model to obtain a classification result. It serves as the baseline strategy
 * and is suitable for most standard classification tasks.
 *
 * <h2>Classification Process</h2>
 * <pre>
 * Input: Question & Answer
 *    ↓
 * Model (single call)
 *    → Prediction: Category Name
 *    → Optional: Explanation, Confidence
 *    ↓
 * Output: Classification Result
 * </pre>
 *
 * <h2>Client Type Handling</h2>
 * The strategy adapts its behavior based on the configured client type:
 * <ul>
 *   <li><b>LocalClient</b>: Serializes the entire ClassificationRequest as JSON
 *       and sends it to a local model endpoint. Useful for custom local models
 *       that expect structured input.</li>
 *   <li><b>Remote Clients</b>: Uses PromptUtils to build a formatted prompt string
 *       from templates defined in ModelProperties. Suitable for API-based models
 *       (OpenAI, Anthropic, etc.).</li>
 * </ul>
 *
 * <h2>Response Format Support</h2>
 * <ul>
 *   <li><b>JSON_OBJECT</b>: Expects structured JSON response that is deserialized
 *       into ClassificationResult. Provides access to label, explanation, and confidence.</li>
 *   <li><b>Plain Text</b>: Expects raw text response containing only the label name.
 *       Creates ClassificationResult with just the name field populated.</li>
 * </ul>
 *
 * <h2>Use Cases</h2>
 * <ul>
 *   <li>Standard classification tasks with single model</li>
 *   <li>Baseline for comparing more complex strategies</li>
 *   <li>Low-latency classification with minimal API calls</li>
 *   <li>Cost-effective classification for large datasets</li>
 * </ul>
 *
 * <h2>Performance Characteristics</h2>
 * <ul>
 *   <li>Single API call per classification</li>
 *   <li>Lowest latency among all strategies</li>
 *   <li>Most cost-effective approach</li>
 * </ul>
 *
 * @see ClassificationRequest
 * @see ClassificationResult
 * @see ModelProperties
 */
@Node(label = "SingleStrategy")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class SingleStrategy extends Neo4jNode implements ClassificationStrategy {
    /**
     * Model configuration to use for classification.
     * <p>
     * This configuration supplies the client, prompt template, response format
     * and other parameters required to execute a single classification.
     */
    @JsonProperty("model")
    @JsonPropertyDescription("The model configuration to use for classification.")
    @Neo4jIgnore
    private ModelProperties model;

    /**
     * Executes single-call classification using the configured model.
     * <p>
     * Behavior varies based on client type and response format:
     * <ul>
     *   <li><b>LocalClient</b>: Serializes entire ClassificationRequest and sends to local endpoint</li>
     *   <li><b>Remote Client + JSON_OBJECT</b>: Builds prompt, sends to API, deserializes JSON response</li>
     *   <li><b>Remote Client + Plain Text</b>: Builds prompt, sends to API, uses raw text as label</li>
     * </ul>
     *
     * @param request the classification request containing text and taxonomy
     * @return ClassificationResult with predicted label and available metadata
     *         (explanation and confidence for JSON responses, label only for plain text)
     */
    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        String prompt = switch (model.getClient()) {
            case LocalClient ignore -> SerializationUtils.serialize(request);
            default -> PromptUtils.replacePrompt(request, model, ClassificationResult.JSON_SCHEME);
        };

        if (model.getResponseFormat() == ModelProperties.ResponseFormat.JSON_OBJECT) {
            return model.getClient()
                        .makeRequest(prompt, ClassificationResult.class);
        } else {
            String response = model.getClient()
                                   .makeRequest(prompt);
            return ClassificationResult.builder().name(response).build();
        }
    }

    @AfterDeserialization
    private void createNode() {
        String query = """
                MATCH(n:%s)-[:%s]->(m:%s)
                WHERE elementId(m) = $modelPropertiesId
                RETURN n
                """.formatted(Neo4jNode.getLabel(SingleStrategy.class),
                              Neo4jRelation.getType(HasClassificationModel.class),
                              Neo4jNode.getLabel(ModelProperties.class));

        SingleStrategy existingNode = GlobalConfig.NEO4J_CLIENT.executeQuery(query, Map.of("modelPropertiesId",
                                                                                           model.getElementId()),
                                                                             SingleStrategy.class).stream().findFirst()
                                                               .orElse(null);

        if (existingNode != null && allRelationsExist(existingNode)) {
            setElementId(existingNode.getElementId());
            return;
        }

        GlobalConfig.NEO4J_CLIENT.saveNode(this);
        createRelationIfNeeded(model, HasClassificationModel.builder().build());
    }

    private boolean allRelationsExist(SingleStrategy existingNode) {
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
