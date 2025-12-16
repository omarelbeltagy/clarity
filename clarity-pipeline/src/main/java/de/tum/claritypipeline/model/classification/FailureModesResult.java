package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import lombok.*;

import java.io.Serializable;
import java.util.List;

/**
 * Structured analysis emitted by the prompt-enhancing diagnose step.
 * <p>
 * Describes high-level failure modes observed across recent misclassifications and links them
 * to concrete prompt fragments that likely caused the issue. The enhancer feeds this object
 * into subsequent patch prompts to guide LLM-generated improvements.
 * </p>
 */
@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class FailureModesResult implements Serializable {
    public static final String JSON_SCHEME = """
            {
                "name": <STRING | Name of the failure mode>,
                "description": <STRING | Description of the failure mode>,
                "prompt_drivers": [
                    {
                        "exact_or_paraphrased_line": <STRING | Exact or paraphrased line from the prompt>,
                        "why_it_matters": <STRING | Explanation why this line is a driver for the failure mode>
                    }
                ]
            }
            """;
    @JsonProperty("failure_modes")
    @JsonPropertyDescription("List of identified failure modes describing recurring misclassification patterns")
    private List<FailureMode> failureModes;

    /**
     * Individual failure mode entry containing human-readable diagnostics and prompt drivers.
     */
    @Getter
    @Setter
    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    public static class FailureMode {
        @JsonProperty("name")
        @JsonPropertyDescription("Short label describing the failure mode (e.g., 'Ambiguous instructions')")
        private String name;

        @JsonProperty("description")
        @JsonPropertyDescription("Detailed explanation of the failure mode and when it manifests")
        private String description;

        @JsonProperty("prompt_drivers")
        @JsonPropertyDescription("Prompt excerpts or paraphrases that likely triggered this failure")
        private List<PromptDriver> promptDrivers;

        /**
         * Pinpoints a specific prompt line and why it contributes to the failure mode.
         */
        @Getter
        @Setter
        @Builder
        @AllArgsConstructor
        @NoArgsConstructor
        public static class PromptDriver {
            @JsonProperty("exact_or_paraphrased_line")
            @JsonPropertyDescription("Problematic prompt line (exact or paraphrased) linked to the failure mode")
            private String exactOrParaphrasedLine;

            @JsonProperty("why_it_matters")
            @JsonPropertyDescription("Reasoning why the referenced line drives the observed failure")
            private String whyItMatters;
        }
    }
}
