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
     * Execute the single-call classification.
     * <p>
     * Behavior:
     * - If the configured client is a LocalClient, the method serializes the
     * entire ClassificationRequest and hands it to the client.
     * - Otherwise, it calls PromptUtils.replacePrompt to produce the prompt
     * string to send to the remote client.
     * - If the model expects JSON_OBJECT output the method deserializes the
     * response into a ClassificationResult. For non-JSON responses the raw
     * string is used as the predicted label name.
     *
     * @param request the classification request containing text and taxonomy.
     * @return a ClassificationResult representing the predicted class and any
     * available metadata (explanation, confidence). For plain text
     * responses the result will have only the name field populated.
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
