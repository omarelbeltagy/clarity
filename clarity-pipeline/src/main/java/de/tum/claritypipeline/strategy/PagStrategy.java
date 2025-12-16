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
 * Paraphrase-Augmented Generation (PAG) classification strategy.
 * <p>
 * This strategy improves classification robustness by generating paraphrases of the input question
 * and aggregating predictions across multiple semantically equivalent formulations. The approach
 * helps reduce sensitivity to specific phrasings and can improve accuracy on ambiguous inputs.
 *
 * <h2>Classification Process</h2>
 * <pre>
 * Input: Question & Answer
 *    ↓
 * Check for Existing Paraphrases in Neo4j
 *    ↓
 * If needed: Paraphrasing Model
 *    → Paraphrase 1: "What is...?"
 *    → Paraphrase 2: "Can you explain...?"
 *    → Paraphrase 3: "How would you describe...?"
 *    ↓
 * Store New Paraphrases in Neo4j
 *    ↓
 * Classification Model (for each paraphrase)
 *    → Prediction 1: Category A
 *    → Prediction 2: Category A
 *    → Prediction 3: Category B
 *    ↓
 * Majority Vote Aggregation
 *    ↓
 * Output: Category A (most common)
 * </pre>
 *
 * <h2>Paraphrase Caching</h2>
 * Paraphrases are stored in Neo4j with relationships:
 * <ul>
 *   <li>QA --[HAS_PARAPHRASE]→ Paraphrase --[PARAPHRASED_BY]→ ModelProperties</li>
 * </ul>
 * This enables:
 * <ul>
 *   <li>Reuse of paraphrases across classification runs</li>
 *   <li>Tracking which model generated each paraphrase</li>
 *   <li>Reduced API costs by avoiding regeneration</li>
 * </ul>
 *
 * <h2>Aggregation Strategy</h2>
 * Uses simple majority voting to select the most frequently predicted label across
 * all paraphrased versions. In case of ties, returns the first label that achieved
 * the maximum count.
 *
 * <h2>Use Cases</h2>
 * <ul>
 *   <li>Improving robustness to input phrasing variations</li>
 *   <li>Handling ambiguous or poorly-formulated questions</li>
 *   <li>Reducing model sensitivity to specific word choices</li>
 *   <li>Ensemble-like behavior without multiple models</li>
 * </ul>
 *
 * <h2>Requirements</h2>
 * <ul>
 *   <li>Paraphrasing model must use JSON_OBJECT format (to return structured list)</li>
 *   <li>LocalClient is not supported for paraphrasing</li>
 *   <li>Classification model can use any format</li>
 * </ul>
 *
 * <h2>Performance Considerations</h2>
 * <ul>
 *   <li>First run: (k paraphrases) + (k classifications) API calls</li>
 *   <li>Subsequent runs: k classification calls (paraphrases cached)</li>
 *   <li>Consider k=3-5 for balance between robustness and cost</li>
 * </ul>
 *
 * @see Paraphrase
 * @see ParaphrasingResult
 * @see ClassificationResult
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

    /**
     * Model configuration for classifying each paraphrased question.
     * Can use any client type and response format.
     */
    @JsonProperty("classification-model")
    @JsonPropertyDescription("The model configuration to use for classification.")
    @Neo4jIgnore
    private ModelProperties classificationModel;

    /**
     * Model configuration for generating question paraphrases.
     * Must support JSON response format to return structured list of paraphrases.
     */
    @JsonProperty("paraphrasing-model")
    @JsonPropertyDescription("The model configuration to use for paraphrasing.")
    @Neo4jIgnore
    private ModelProperties paraphrasingModel;

    /**
     * Number of paraphrases to generate and classify.
     * Higher values provide more robust aggregation but increase API costs.
     * Typical values: 3-5.
     */
    @JsonProperty("k")
    @JsonPropertyDescription("The number of paraphrases to generate and use")
    private int k = 3;

    /**
     * Executes the paraphrase-augmented classification strategy.
     * <p>
     * Workflow:
     * <ol>
     *   <li>Validates configuration (JSON format for paraphrasing, not LocalClient)</li>
     *   <li><b>Paraphrase Retrieval/Generation</b>:
     *     <ul>
     *       <li>Queries Neo4j for existing paraphrases from this paraphrasing model</li>
     *       <li>If fewer than k exist, generates new paraphrases to reach k</li>
     *       <li>Stores new paraphrases in Neo4j with proper relationships</li>
     *     </ul>
     *   </li>
     *   <li><b>Classification Phase</b>:
     *     <ul>
     *       <li>For each of the k paraphrases:</li>
     *       <li>Creates classification request with paraphrased question</li>
     *       <li>Classifies using classification model</li>
     *       <li>Collects all classification results</li>
     *     </ul>
     *   </li>
     *   <li><b>Aggregation</b>:
     *     <ul>
     *       <li>Applies majority voting to select most common label</li>
     *       <li>Returns final classification result</li>
     *     </ul>
     *   </li>
     * </ol>
     *
     * @param request the classification request with original question and answer
     * @return aggregated classification result from majority vote
     * @throws RuntimeException      if no paraphrases can be generated
     * @throws IllegalStateException if no classification results obtained
     */
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

    /**
     * Determines the most frequently occurring label across paraphrase classifications.
     * <p>
     * Uses simple frequency counting. In case of ties, returns the first label
     * that achieved the maximum count during iteration.
     *
     * @param labels list of predicted labels from paraphrase classifications
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
     * Validates that the configuration supports PAG strategy requirements.
     * <p>
     * Ensures:
     * <ul>
     *   <li>Paraphrasing model is not LocalClient</li>
     *   <li>Paraphrasing model uses JSON_OBJECT format (required for structured output)</li>
     * </ul>
     *
     * @throws UnsupportedOperationException if any requirement is not met
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
