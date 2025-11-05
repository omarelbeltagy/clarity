package de.tum.claritypipeline.model.properties;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Enumeration of supported response formats returned by a classifier.
 *
 * <p>- JSON_OBJECT: structured JSON response that can be parsed into fields.<br>
 * - TEXT: plain text response that may require regex extraction.
 */
public enum ResponseFormat {
    /**
     * Indicates the classifier returns a structured JSON object.
     *
     * <p>Serialized as "json_object".
     */
    @JsonProperty("json_object")
    JSON_OBJECT,

    /**
     * Indicates the classifier returns plain textual output.
     *
     * <p>Serialized as "text".
     */
    @JsonProperty("text")
    TEXT
}