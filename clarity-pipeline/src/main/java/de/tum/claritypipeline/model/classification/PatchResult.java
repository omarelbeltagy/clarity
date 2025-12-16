package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonProperty;
import de.tum.claritypipeline.model.core.Taxonomy;
import lombok.*;

import java.io.Serializable;
import java.util.List;

@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class PatchResult implements Serializable {
    public static final String JSON_SCHEME = """
            {
                "revised_prompt": <STRING | The revised prompt after applying the patches>,
                "revised_taxonomy": [
                   {
                        "name": <STRING | Name of the category>,
                        "description": <STRING | Description of the category>,
                        "examples": [
                            {
                                "question": <STRING | Example question>,
                                "answer": <STRING | Example answer>
                                "explanation": <STRING | Explanation for the labeling>
                            }
                        ]
                    },
                    ...
                ],
                "patch_notes": [
                    <STRING | Note describing a specific change made in the patch>,
                    ...
                ]
            }
            """;
    @JsonProperty("patch_notes")
    private List<String> patchNotes;
    @JsonProperty("revised_prompt")
    private String revisedPrompt;
    @JsonProperty("revised_taxonomy")
    private List<Taxonomy.Category> revisedTaxonomy;
}
