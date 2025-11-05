package de.tum.claritypipeline.model.config;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Configuration holder for a regex pattern used to extract labels from client text responses.
 *
 * <p>Contains the regex string and a pipe-separated list of human-friendly flag names which are mapped
 * to {@link Pattern} constants by {@link #getFlagsMask()}.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class PatternProperties {

    /**
     * Mapping from human-friendly flag names to java.util.regex.Pattern flags.
     *
     * <p>Keys are normalized (lowercase, hyphenated) versions of user-provided flag names.
     */
    private static final Map<String, Integer> FLAG_MAPPINGS = Map.of(
            "case-insensitive", Pattern.CASE_INSENSITIVE,
            "multiline", Pattern.MULTILINE,
            "dotall", Pattern.DOTALL,
            "unicode-case", Pattern.UNICODE_CASE,
            "canon-eq", Pattern.CANON_EQ,
            "unix-lines", Pattern.UNIX_LINES,
            "literal", Pattern.LITERAL,
            "unicode-character-class", Pattern.UNICODE_CHARACTER_CLASS,
            "comments", Pattern.COMMENTS
    );

    /**
     * The regular expression used to extract the label from a textual response.
     *
     * <p>Default: "^Label:\s*(.+)$"
     */
    @JsonProperty("regex")
    @JsonPropertyDescription("The regex pattern to extract labels from text responses.")
    private String regex = "^Label:\\s*(.+)$";

    /**
     * Pipe-separated list of flag names to enable for the regex, e.g. "multiline|case-insensitive".
     *
     * <p>Default: "multiline"
     */
    @JsonProperty("flags")
    @JsonPropertyDescription("The regex flags to use, separated by '|'. E.g., 'multiline|case-insensitive'.")
    private String flags = "multiline";

    /**
     * Convert the human-friendly flags string into an integer mask appropriate for {@link Pattern}.
     *
     * <p>If flags is null or empty, {@link Pattern#MULTILINE} is returned by default.
     *
     * @return combined int mask of Pattern flags
     */
    public int getFlagsMask() {
        if (flags == null || flags.isEmpty()) {
            return Pattern.MULTILINE;
        }

        return Arrays.stream(flags.split("\\|"))
                     .map(String::trim)
                     .map(f -> f.toLowerCase().replace("_", "-"))
                     .map(FLAG_MAPPINGS::get)
                     .filter(Objects::nonNull)
                     .reduce(0, (a, b) -> a | b);
    }
}