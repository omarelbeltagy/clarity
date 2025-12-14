package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import lombok.*;

import java.io.Serializable;
import java.util.List;

@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@Builder
public class BestGuessClassificationResult implements Serializable {

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
    @JsonProperty("top_labels")
    @JsonPropertyDescription("Most likely classification categories")
    private List<ClassificationResult> topLabels;
}
