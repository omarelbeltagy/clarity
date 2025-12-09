package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonIgnore;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.model.core.Taxonomy;
import lombok.*;

import java.io.Serializable;

/**
 * Request payload for a single classification operation.
 *
 * <p>Contains the textual question and optional contextual information that the classifier may use.
 */
@Getter
@Setter
@Builder(toBuilder = true)
@AllArgsConstructor
@NoArgsConstructor
public class ClassificationRequest implements Serializable {

    /**
     * The main question or text to classify.
     *
     * <p>Typically a short sentence or question.
     */
    private String question;

    /**
     * Optional additional context that may influence classification.
     *
     * <p>May be null or empty if not needed.
     */
    private String context;

    /**
     * The answer associated with the question, if applicable.
     * Only used for internal classifier logic or evaluation. Not serialized.
     *
     * <p>Ignored during serialization.
     */
    @JsonIgnore
    private String answer;

    /**
     * The taxonomy to use for classification.
     * Only used for internal classifier logic. Not serialized.
     *
     * <p>Ignored during serialization.
     */
    @JsonIgnore
    private Taxonomy taxonomy;

    /**
     * The full QA pair
     *
     * <p>Ignored during serialization.
     */
    @JsonIgnore
    private QA qa;
}
