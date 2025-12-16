package de.tum.claritypipeline.model.core;

import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import lombok.*;

/**
 * Represents a question-answer item, e.g. extracted from an interview or dataset.
 *
 * <p>Contains original and derived fields, annotations and metadata used for classification/evaluation.
 * </p>
 * Stores a single paraphrased question produced/consumed by {@link de.tum.claritypipeline.strategy.PagStrategy}.
 * <p>
 * Paraphrases are persisted in Neo4j so future runs can reuse them without re-calling the paraphrasing model.
 * Each node keeps the paraphrased question text plus provenance via {@code HasParaphrase}/{@code ParaphrasedBy}.
 * </p>
 */
@Node(label = "Paraphrase")
@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class Paraphrase extends Neo4jNode {
    private String question;
}