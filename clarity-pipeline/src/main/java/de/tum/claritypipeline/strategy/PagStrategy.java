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
import de.tum.claritypipeline.model.classification.ParaphrasingResult;
import de.tum.claritypipeline.model.config.GlobalConfig;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.model.core.Paraphrase;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.model.relation.HasClassificationModel;
import de.tum.claritypipeline.model.relation.HasParaphrase;
import de.tum.claritypipeline.model.relation.HasParaphrasingModel;
import de.tum.claritypipeline.model.relation.ParaphrasedBy;
import de.tum.claritypipeline.utils.PromptUtils;
import de.tum.clarityutils.AfterDeserialization;
import de.tum.clarityutils.SerializationUtils;
import lombok.*;
import org.slf4j.Logger;

import java.util.ArrayList;
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
@Node(label = "PAGStrategy")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class PagStrategy extends Neo4jNode implements ClassificationStrategy {
    @JsonIgnore
    @Neo4jIgnore
    private final Logger log = org.slf4j.LoggerFactory.getLogger(PagStrategy.class);

    @JsonProperty("classification-model")
    @JsonPropertyDescription("The model configuration to use for classification.")
    @Neo4jIgnore
    private ModelProperties classificationModel;

    @JsonProperty("paraphrasing-model")
    @JsonPropertyDescription("The model configuration to use for paraphrasing.")
    @Neo4jIgnore
    private ModelProperties paraphrasingModel;

    @JsonProperty("k")
    @JsonPropertyDescription("The number of paraphrases to generate and use")
    private int k = 3;

    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        validateConfiguration();
        String existingParaphrasesQuery = """
                MATCH(qa:%s)-[:%s]->(n:%s)-[%s]->(pm:%s)
                WHERE elementId(qa) = $qaId
                AND elementId(pm) = $paraphrasingModelPropertiesId
                RETURN n
                """.formatted(
                Neo4jNode.getLabel(QA.class),
                Neo4jRelation.getType(HasParaphrase.class),
                Neo4jNode.getLabel(Paraphrase.class),
                Neo4jRelation.getType(ParaphrasedBy.class),
                Neo4jNode.getLabel(ModelProperties.class)
        );
        List<Paraphrase> existingParaphrases = GlobalConfig.NEO4J_CLIENT.executeQuery(existingParaphrasesQuery,
                                                                                      Map.of("qaId", request.getQa()
                                                                                                            .getElementId(),
                                                                                             "paraphrasingModelPropertiesId",
                                                                                             paraphrasingModel.getElementId()),
                                                                                      Paraphrase.class);
        int needed = k - existingParaphrases.size();
        if (needed > 0) {
            log.info("Generating {} new paraphrases for QA ({}) with {}", needed, request.getQa().getQuestion(),
                     paraphrasingModel.getName());
            String paraphrasingPrompt = PromptUtils.replacePrompt(request, paraphrasingModel,
                                                                  ParaphrasingResult.JSON_SCHEME)
                                                   .replace("{k}", String.valueOf(needed));
            ParaphrasingResult paraphrases = paraphrasingModel.getClient().makeRequest(paraphrasingPrompt,
                                                                                       ParaphrasingResult.class);
            if (paraphrases == null || paraphrases.getParaphrases() == null || paraphrases.getParaphrases().isEmpty()) {
                throw new RuntimeException(
                        "No new paraphrases generated for QA (" + request.getQa().getQuestion() + ")");
            }
            for (String paraphrase : paraphrases.getParaphrases()) {
                Paraphrase newParaphrase = Paraphrase.builder().question(paraphrase).build();
                GlobalConfig.NEO4J_CLIENT.saveNode(newParaphrase);

                HasParaphrase hasParaphraseRel = new HasParaphrase();
                hasParaphraseRel.setStartNodeId(request.getQa().getElementId());
                hasParaphraseRel.setEndNodeId(newParaphrase.getElementId());

                ParaphrasedBy paraphrasedByRel = new ParaphrasedBy();
                paraphrasedByRel.setStartNodeId(newParaphrase.getElementId());
                paraphrasedByRel.setEndNodeId(paraphrasingModel.getElementId());

                GlobalConfig.NEO4J_CLIENT.createRelation(hasParaphraseRel);
                GlobalConfig.NEO4J_CLIENT.createRelation(paraphrasedByRel);

                existingParaphrases.add(newParaphrase);
            }
        }
        if (existingParaphrases.size() > k) {
            existingParaphrases = existingParaphrases.stream().limit(k).toList();
        }

        List<ClassificationResult> classificationResults = new ArrayList<>();

        for (Paraphrase paraphrase : existingParaphrases) {
            ClassificationRequest cr = request.toBuilder()
                                              .qa(request.getQa().toBuilder().build())
                                              .build();
            cr.getQa().setQuestion(paraphrase.getQuestion());
            cr.setQuestion(paraphrase.getQuestion());

            String classificationPrompt = switch (classificationModel.getClient()) {
                case LocalClient ignore -> SerializationUtils.serialize(cr);
                default -> PromptUtils.replacePrompt(cr, classificationModel, ClassificationResult.JSON_SCHEME);
            };

            ClassificationResult result;
            if (classificationModel.getResponseFormat() == ModelProperties.ResponseFormat.JSON_OBJECT) {
                result = classificationModel.getClient()
                                            .makeRequest(classificationPrompt, ClassificationResult.class);
            } else {
                String response = classificationModel.getClient()
                                                     .makeRequest(classificationPrompt);
                result = ClassificationResult.builder().name(response).build();
            }
            if (result != null) {
                classificationResults.add(result);
            }
        }

        if (classificationResults.isEmpty()) {
            throw new IllegalStateException("No classification results obtained from paraphrases");
        }

        List<String> labels = classificationResults.stream()
                                                   .map(ClassificationResult::getName)
                                                   .toList();

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
        if (paraphrasingModel.getClient() instanceof LocalClient) {
            throw new UnsupportedOperationException(
                    "LocalClient is not supported for Paraphrasing"
            );
        }

        if (paraphrasingModel.getResponseFormat() != ModelProperties.ResponseFormat.JSON_OBJECT) {
            throw new UnsupportedOperationException(
                    "Only ResponseFormat.JSON_OBJECT is supported for the Paraphrasing Model in " +
                            "PAGStrategy, because the Model needs a JSON response format"
            );
        }
    }

    @AfterDeserialization
    private void createNode() {
        String query = """
                MATCH(pm:%s)<-[:%s]-(n:%s)-[:%s]->(cm:%s)
                WHERE elementId(cm) = $classificationModelPropertiesId
                AND elementId(pm) = $paraphrasingModelPropertiesId
                AND n.k = $k
                RETURN n
                """.formatted(
                Neo4jNode.getLabel(ModelProperties.class),
                Neo4jRelation.getType(HasParaphrasingModel.class),
                Neo4jNode.getLabel(PagStrategy.class),
                Neo4jRelation.getType(HasClassificationModel.class),
                Neo4jNode.getLabel(ModelProperties.class));

        PagStrategy existingNode = GlobalConfig.NEO4J_CLIENT.executeQuery(query,
                                                                          Map.of("classificationModelPropertiesId",
                                                                                 classificationModel.getElementId(),
                                                                                 "paraphrasingModelPropertiesId",
                                                                                 paraphrasingModel.getElementId(),
                                                                                 "k",
                                                                                 k),
                                                                          PagStrategy.class).stream()
                                                            .findFirst()
                                                            .orElse(null);

        if (existingNode != null && allRelationsExist(existingNode)) {
            setElementId(existingNode.getElementId());
            return;
        }

        GlobalConfig.NEO4J_CLIENT.saveNode(this);
        createRelationIfNeeded(classificationModel, HasClassificationModel.builder().build());
        createRelationIfNeeded(paraphrasingModel, HasParaphrasingModel.builder().build());
    }

    private boolean allRelationsExist(PagStrategy existingNode) {
        boolean classificationModelRelationOk =
                GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(), classificationModel.getElementId(),
                                                       HasClassificationModel.class)
                        != null;
        boolean paraphrasingModelRelationOk =
                GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(), paraphrasingModel.getElementId(),
                                                       HasParaphrasingModel.class)
                        != null;
        return classificationModelRelationOk && paraphrasingModelRelationOk;
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
