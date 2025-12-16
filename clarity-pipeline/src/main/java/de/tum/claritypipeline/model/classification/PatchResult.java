package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.claritypipeline.model.core.Taxonomy;
import lombok.*;

import java.io.Serializable;
import java.util.List;

/**
 * Result object produced by the patch phase of {@link de.tum.claritypipeline.service.PromptEnhancer}.
 * <p>
 * Captures the revised prompt text, refined taxonomy snapshot, and human-readable patch notes so
 * that teams can inspect, validate, and optionally persist changes back into configuration files.
 * </p>
 */
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
    @JsonPropertyDescription("Narrative list explaining each prompt or taxonomy adjustment that was applied")
    private List<String> patchNotes;

    @JsonProperty("revised_prompt")
    @JsonPropertyDescription("Updated prompt template incorporating all accepted fixes")
    private String revisedPrompt;

    @JsonProperty("revised_taxonomy")
    @JsonPropertyDescription("Revised taxonomy snapshot (same category set, refined descriptions/examples only)")
    private List<Taxonomy.Category> revisedTaxonomy;
}
