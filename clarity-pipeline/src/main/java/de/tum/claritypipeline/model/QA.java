package de.tum.claritypipeline.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import lombok.*;

/**
 * Represents a question-answer item, e.g. extracted from an interview or dataset.
 *
 * <p>Contains original and derived fields, annotations and metadata used for classification/evaluation.
 */
@Node(label = "QA")
@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class QA extends Neo4jNode {

    /**
     * Sequential index of the QA item in the source dataset.
     */
    @JsonProperty("index")
    private int index;

    /**
     * The interview question as recorded.
     */
    @JsonProperty("interview_question")
    private String interviewQuestion;

    /**
     * The corresponding interview answer.
     */
    @JsonProperty("interview_answer")
    private String interviewAnswer;

    /**
     * The processed question used for classification.
     */
    @JsonProperty("question")
    private String question;

    /**
     * The processed answer or title field.
     */
    @JsonProperty("title")
    private String title;

    /**
     * Date associated with the QA item (ISO string or similar).
     */
    @JsonProperty("date")
    private String date;

    /**
     * Name of the president or subject referenced.
     */
    @JsonProperty("president")
    private String president;

    /**
     * Source URL for the QA item.
     */
    @JsonProperty("url")
    private String url;

    /**
     * Order of the question within a larger set.
     */
    @JsonProperty("question_order")
    private int questionOrder;

    /**
     * Model-generated summary using GPT-3.5 (if available).
     */
    @JsonProperty("gpt3.5_summary")
    private String gpt3_5Summary;

    /**
     * Model-generated prediction using GPT-3.5 (if available).
     */
    @JsonProperty("gpt3.5_prediction")
    private String gpt3_5Prediction;

    /**
     * Identifier of the annotator who provided labels.
     */
    @JsonProperty("annotator_id")
    private String annotatorId;

    /**
     * Label or note from annotator 1.
     */
    @JsonProperty("annotator1")
    private String annotator1;

    /**
     * Label or note from annotator 2.
     */
    @JsonProperty("annotator2")
    private String annotator2;

    /**
     * Label or note from annotator 3.
     */
    @JsonProperty("annotator3")
    private String annotator3;

    /**
     * Whether the audio/text was marked as inaudible.
     */
    @JsonProperty("inaudible")
    private boolean inaudible;

    /**
     * Whether multiple questions were detected.
     */
    @JsonProperty("multiple_questions")
    private boolean multipleQuestions;

    /**
     * Whether the questions are affirmative in nature.
     */
    @JsonProperty("affirmative_questions")
    private boolean affirmativeQuestions;

    /**
     * Gold or annotated clarity label.
     */
    @JsonProperty("clarity_label")
    private String clarityLabel;

    /**
     * Gold or annotated evasion label.
     */
    @JsonProperty("evasion_label")
    private String evasionLabel;

    /**
     * Whether this item is part of a test set.
     */
    @JsonProperty("test")
    private boolean test;

    /**
     * Whether the item is considered valid.
     */
    @JsonProperty("valid")
    private boolean valid;

    /**
     * Whether the item is used for training.
     */
    @JsonProperty("train")
    private boolean train;
}