package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import lombok.*;

import java.io.Serializable;
import java.util.Map;

/**
 * Represents the result of a classification operation for a single instance.
 *
 * <p>Holds the chosen category name, an explanation for the decision,
 * a confidence score and optional per-category scores that are not persisted in Neo4j.
 * <p>
 * Canonical representation of a single prediction stored in Neo4j and exported via the pipeline.
 * <p>
 * Captures the selected taxonomy category, a free-form explanation, confidence score,
 * and optional runtime metadata (per-class scores, judgement annotations). Instances are
 * created by all strategies, persisted through {@link de.tum.claritypipeline.service.ClassificationPipeline},
 * and later consumed by evaluators, exporters, or prompt enhancers.
 * </p>
 *
 * <p>Neo4j persistence (through {@link de.tum.clarityneo4j.core.Neo4jNode}) ensures end-to-end provenance
 * linking QA nodes, strategy configurations, and evaluation artifacts.</p>
 */
@Node(label = "ClassificationResult")
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@Builder
public class ClassificationResult extends Neo4jNode implements Serializable {
    public static final String JSON_SCHEME = """
            {
              "name": <STRING | Name of the classified category>,
              "explanation": <STRING | Brief, text-grounded justification citing the key phrase(s) or absence of an answer>,
              "confidence": <DOUBLE | Confidence score between 0 and 1 inclusive, up to two decimals.>
            }
            """;
    @JsonIgnore
    private String demandSlot;
    @JsonIgnore
    private Boolean settled;
    @JsonIgnore
    private String settledValueOneClause;
    @JsonIgnore
    private String evidenceQuote;
    @JsonIgnore
    private String avoidanceType;

    /**
     * A short textual explanation why this category was assigned.
     *
     * <p>Helps with interpretability and auditing of the classifier output.
     */
    @JsonProperty("explanation")
    @JsonPropertyDescription("Explanation for the classification decision")
    private String explanation;
    /**
     * The name of the predicted category.
     *
     * <p>Serialized as "name" in JSON/YAML.
     */
    @JsonProperty("name")
    @JsonPropertyDescription("Name of the classified category")
    private String name;

    /**
     * Optional map of category -> score for all considered categories.
     *
     * <p>This field is ignored for persistence/serialization and intended for runtime diagnostics.
     */
    @JsonIgnore
    @Neo4jIgnore
    private Map<String, Double> scores;

    /**
     * If the classification strategy is 'judgement' this is the explanation provided by the judgement model that
     * verifies the initial classification.
     *
     * <p>Ignored for persistence/serialization.
     */
    @JsonIgnore
    private String judgementExplanation;

    /**
     * If the classification strategy is 'judgement' this is the confidence score provided by the judgement model
     * that verifies the initial classification.
     *
     * <p>Ignored for persistence/serialization.
     */
    @JsonIgnore
    private Double judgementConfidence;
    /**
     * A confidence score for the selected category, typically in the range [0.0, 1.0].
     *
     * <p>This represents the classifier's confidence in the chosen label.
     */
    @JsonProperty("confidence")
    @JsonPropertyDescription("Confidence score between 0 and 1 inclusive, up to two decimals.")
    private Double confidence;
}
