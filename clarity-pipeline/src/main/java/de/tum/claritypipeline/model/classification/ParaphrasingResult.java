package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import lombok.*;

import java.io.Serializable;
import java.util.List;

/**
 * Request payload for a single classification operation.
 *
 * <p>Contains the textual question and optional contextual information that the classifier may use.
 */
@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class ParaphrasingResult implements Serializable {

    public static final String JSON_SCHEME = """
            {
                "paraphrases": [
                    <STRING | Paraphrase for the input>
                ]
            }
            """;
    @JsonProperty("paraphrases")
    @JsonPropertyDescription("The list of paraphrases for the input")
    private List<String> paraphrases;
}
