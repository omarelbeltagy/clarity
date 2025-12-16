package de.tum.claritypipeline.model.config;

import com.fasterxml.jackson.annotation.JsonProperty;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.claritypipeline.model.core.QA;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Enumeration of embedding indices used in the application.
 *
 * <p>Defines the different types of embedding indices along with their associated properties.
 */
@Getter
@AllArgsConstructor
public enum EmbeddingIndex {
    @JsonProperty("qa_answer")
    QA_ANSWER("qa_answer_embeddings", "questionEmbedding", QA.class),

    @JsonProperty("qa_question")
    QA_QUESTION("qa_question_embeddings", "answerEmbedding", QA.class),

    @JsonProperty("qa_question_and_answer")
    QA_QUESTION_AND_ANSWER("qa_question_and_answer_embeddings", "questionAnswerEmbedding", QA.class);

    private final String indexName;
    private final String fieldName;
    private final Class<? extends Neo4jNode> nodeClass;
}
