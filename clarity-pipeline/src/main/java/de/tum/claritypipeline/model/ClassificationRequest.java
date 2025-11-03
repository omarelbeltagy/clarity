package de.tum.claritypipeline.model;

import lombok.*;

import java.io.Serializable;

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
public class ClassificationRequest implements Serializable {

    /**
     * The main question or text to classify.
     *
     * <p>Typically a short sentence or question.
     */
    String question;

    /**
     * Optional additional context that may influence classification.
     *
     * <p>May be null or empty if not needed.
     */
    String context;
}
