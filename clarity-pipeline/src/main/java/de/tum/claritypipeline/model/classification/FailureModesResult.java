package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.io.Serializable;
import java.util.List;

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
    private List<FailureMode> failureModes;

    @Getter
    @Setter
    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    public static class FailureMode {
        private String name;
        private String description;

        @JsonProperty("prompt_drivers")
        private List<PromptDriver> promptDrivers;

        @Getter
        @Setter
        @Builder
        @AllArgsConstructor
        @NoArgsConstructor
        public static class PromptDriver {
            @JsonProperty("exact_or_paraphrased_line")
            private String exactOrParaphrasedLine;
            @JsonProperty("why_it_matters")
            private String whyItMatters;
        }
    }
}
