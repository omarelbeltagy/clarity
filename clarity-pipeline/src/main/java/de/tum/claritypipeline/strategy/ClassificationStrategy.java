package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;

/**
 * Strategy interface for executing a classification workflow.
 * <p>
 * Implementations encapsulate different classification flows (for example:
 * single-call classification or a two-step classification with a judgement
 * phase). Implementations should be serializable/deserializable via Jackson
 * and are discriminated using the "type" property as configured above.
 * <p>
 * Implementations must be thread-safe if instances are reused across threads.
 */
@JsonTypeInfo(
        use = JsonTypeInfo.Id.NAME,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type"
)
@JsonSubTypes({
        @JsonSubTypes.Type(value = SingleStrategy.class, name = "single"),
        @JsonSubTypes.Type(value = JudgementStrategy.class, name = "judgement"),
        @JsonSubTypes.Type(value = DiscussionStrategy.class, name = "discussion"),
        @JsonSubTypes.Type(value = MultiStrategy.class, name = "multi"),
        @JsonSubTypes.Type(value = BestGuessStrategy.class, name = "best-guess"),
        @JsonSubTypes.Type(value = PagStrategy.class, name = "pag")
})
public interface ClassificationStrategy {
    /**
     * Execute the classification strategy for the provided request.
     * <p>
     * Implementations should use the information contained in the
     * ClassificationRequest to interact with configured model clients and
     * produce a ClassificationResult. Implementations may throw runtime
     * exceptions if required configuration is missing or unsupported.
     *
     * @param request non-null request object containing the input text,
     *                taxonomy and any other contextual data required for
     *                producing a classification.
     * @return a ClassificationResult representing the chosen class,
     * optional explanation and confidence scores; never null for a
     * successful execution, but may be null if the strategy decides
     * to indicate failure via a null return (prefer throwing for errors).
     */
    ClassificationResult execute(ClassificationRequest request);

    <T extends Neo4jNode> T getClassificationStrategyNode();
}
