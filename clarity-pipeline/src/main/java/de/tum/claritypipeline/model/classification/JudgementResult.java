package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import lombok.*;

import java.io.Serializable;

/**
 * Represents the result of another LLM judging a prior classification.
 *
 * <p>Holds the (possibly corrected) category name, an explanation for the judgement,
 * a confidence score, and whether the initial classification was confirmed or adjusted.
 */
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@Builder
public class JudgementResult implements Serializable {

    /**
     * The name of the classified category.
     *
     * <p>If the initial classification was incorrect, this is the name of the correct category.
     * Otherwise, it is the name of the initially classified category.
     */
    @JsonProperty("name")
    @JsonPropertyDescription(
            "Name of the classified category. If the initial classification was incorrect, choose the name of the "
                    + "correct category. Otherwise, use the name of the initially classified category.")
    private String name;

    /**
     * A short textual explanation for the judgement decision.
     *
     * <p>Helps with interpretability and auditing of the judgement.
     */
    @JsonProperty("explanation")
    @JsonPropertyDescription("Explanation for the judgement decision")
    private String explanation;

    /**
     * A confidence score for the judgement, typically in the range [0.0, 1.0].
     *
     * <p>If the classification was incorrect, this indicates the confidence in the corrected category.
     * Otherwise, it represents the confidence in the initially classified label.
     */
    @JsonProperty("confidence")
    @JsonPropertyDescription(
            "Confidence score of the classification. If the classification was incorrect, this indicates the "
                    + "confidence in the corrected category.")
    private double confidence;

    /**
     * Indicates whether the initial classification was correct.
     *
     * <p>True if the initial classification was correct; false if it was incorrect and has been adjusted.
     */
    @JsonProperty("confirmed")
    @JsonPropertyDescription("If the initial classification was correct or not")
    private boolean confirmed;

    public static final String JSON_SCHEME = """
            {
                "confirmed": <BOOLEAN | If the initial classification was correct or not>,
                "name": <STRING | Name of the classified category. If the initial classification was incorrect, choose the name of the correct category. Otherwise, use the name of the initially classified category>,
                "explanation": <STRING | Explanation for the judgement decision>,
                "confidence": <DOUBLE | Confidence score of the classification. If the classification was incorrect, this indicates the confidence in the corrected category>
            }
            """;
}
