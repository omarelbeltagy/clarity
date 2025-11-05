package de.tum.claritypipeline.model.properties;

import com.fasterxml.jackson.annotation.JsonProperty;
import de.tum.clarityutils.AfterDeserialization;
import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class RaqProperties {

    @JsonProperty("enabled")
    private boolean enabled;

    @JsonProperty("k")
    private int k = 8;

    @JsonProperty("embedding-index")
    private EmbeddingIndex embeddingIndex;

    @AfterDeserialization
    public void initialize() {
        if (enabled) {
            if (embeddingIndex == null) {
                throw new IllegalArgumentException("RAQ is enabled but embedding index is not set.");
            }
        }
    }
}
