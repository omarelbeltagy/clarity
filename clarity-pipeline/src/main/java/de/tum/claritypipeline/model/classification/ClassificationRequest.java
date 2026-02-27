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
 * Immutable request DTO handed to every {@code ClassificationStrategy}.
 * <p>
 * Bundles the user-facing question, optional interview context, the derived answer snippet, the taxonomy snapshot, and
 * the originating {@link de.tum.claritypipeline.model.core.QA}. Service and strategy layers use this container to pass
 * all required metadata through worker threads, LLM clients, and evaluation routines without leaking persistence
 * objects.
 * </p>
 *
 * <h2>Typical Flow</h2>
 * <ol>
 *   <li>{@link de.tum.claritypipeline.service.ClassificationPipeline} builds the request per QA</li>
 *   <li>Strategy-specific prompts serialize {@code question} and {@code context}</li>
 *   <li>Internal logic may rely on {@code taxonomy}, {@code answer}, or {@code qa}</li>
 * </ol>
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
     * The answer associated with the question, if applicable. Only used for internal classifier logic or evaluation.
     * Not serialized.
     *
     * <p>Ignored during serialization.
     */
    @JsonIgnore
    private String answer;
    
    /**
     * The taxonomy to use for classification. Only used for internal classifier logic. Not serialized.
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
