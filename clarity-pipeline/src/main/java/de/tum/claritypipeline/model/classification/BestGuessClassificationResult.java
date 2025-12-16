package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import lombok.*;

import java.io.Serializable;
import java.util.List;

/**
 * Structured response payload used by {@link de.tum.claritypipeline.strategy.BestGuessStrategy}.
 * <p>
 * Contains the ordered list of top-k {@link ClassificationResult} entries produced by the underlying
 * LLM when asked for multiple candidate labels. The pipeline aggregates these candidates (e.g. via
 * majority vote) to decide on the final category while still keeping full traceability of each guess.
 * </p>
 *
 * <h2>Usage</h2>
 * <ul>
 *   <li>LLM must emit JSON following {@link #JSON_SCHEME}</li>
 *   <li>Entries are processed in order of likelihood/confidence</li>
 *   <li>Downstream components may inspect explanations/confidence per candidate</li>
 * </ul>
 */
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@Builder
public class BestGuessClassificationResult implements Serializable {

    @JsonProperty("top_labels")
    @JsonPropertyDescription("Ordered list of candidate labels (highest confidence first) returned by the model")
    private List<ClassificationResult> topLabels;

    public static final String JSON_SCHEME = """
            {
                "top_labels": [
                    {
                        "name": <STRING | Name of the classified category>,
                        "explanation": <STRING | Explanation for the classification decision>,
                        "confidence": <DOUBLE | Confidence score between 0 and 1 inclusive, up to two decimals.>
                    }
                ]
            }
            """;
}
