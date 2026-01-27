package de.tum.claritypipeline.model.core;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.opencsv.bean.CsvBindByName;
import com.opencsv.bean.CsvIgnore;
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
@Builder(toBuilder = true)
@AllArgsConstructor
@NoArgsConstructor
public class QA extends Neo4jNode {

    /**
     * Sequential index of the QA item in the source dataset.
     * This is unique within the dataset split (train/test) but not globally.
     */
    @JsonProperty("index")
    @CsvBindByName(column = "index")
    @JsonPropertyDescription("Sequential identifier of this QA inside its dataset split.")
    private int index;

    /**
     * The interview question as recorded.
     */
    @JsonProperty("interview_question")
    @CsvBindByName(column = "interview_question")
    @JsonPropertyDescription("Original interview question text as captured in the dataset.")
    private String interviewQuestion;

    /**
     * Cleaned version of the interview question.
     */
    @JsonProperty("interview_question_clean")
    @CsvBindByName(column = "interview_question_clean")
    @JsonPropertyDescription("Normalized/cleaned version of the interview question.")
    private String interviewQuestionClean;

    /**
     * The corresponding interview answer.
     */
    @JsonProperty("interview_answer")
    @CsvBindByName(column = "interview_answer")
    @JsonPropertyDescription("Original interview answer text.")
    private String interviewAnswer;

    /**
     * Cleaned version of the interview answer.
     */
    @JsonProperty("interview_answer_clean")
    @CsvBindByName(column = "interview_answer_clean")
    @JsonPropertyDescription("Normalized/cleaned version of the interview answer.")
    private String interviewAnswerClean;


    /**
     * The processed question used for classification.
     */
    @JsonProperty("question")
    @CsvBindByName(column = "question")
    @JsonPropertyDescription("Processed question string used for prompting the classifier.")
    private String question;

    /**
     * The processed answer or title field.
     */
    @JsonProperty("title")
    @CsvBindByName(column = "title")
    @JsonPropertyDescription("Optional short title or headline associated with the QA entry.")
    private String title;

    /**
     * Date associated with the QA item (ISO string or similar).
     */
    @JsonProperty("date")
    @CsvBindByName(column = "date")
    @JsonPropertyDescription("Date (ISO-like string) when the QA took place.")
    private String date;

    /**
     * Name of the president or subject referenced.
     */
    @JsonProperty("president")
    @CsvBindByName(column = "president")
    @JsonPropertyDescription("Name of the president/person referenced in this QA.")
    private String president;

    /**
     * Source URL for the QA item.
     */
    @JsonProperty("url")
    @CsvBindByName(column = "url")
    @JsonPropertyDescription("Source link for the QA transcript.")
    private String url;

    /**
     * Order of the question within a larger set.
     */
    @JsonProperty("question_order")
    @CsvBindByName(column = "question_order")
    @JsonPropertyDescription("Position of the question within the broader interview/session.")
    private int questionOrder;

    /**
     * Model-generated summary using GPT-3.5 (if available).
     */
    @JsonProperty("gpt3.5_summary")
    @CsvBindByName(column = "gpt3.5_summary")
    @JsonPropertyDescription("Optional GPT-3.5 generated summary when available.")
    private String gpt3_5Summary;

    /**
     * Model-generated prediction using GPT-3.5 (if available).
     */
    @JsonProperty("gpt3.5_prediction")
    @CsvBindByName(column = "gpt3.5_prediction")
    @JsonPropertyDescription("Optional GPT-3.5 baseline prediction stored in the dataset.")
    private String gpt3_5Prediction;

    /**
     * Identifier of the annotator who provided labels.
     */
    @JsonProperty("annotator_id")
    @CsvBindByName(column = "annotator_id")
    @JsonPropertyDescription("Identifier of the human annotator who labeled this QA.")
    private String annotatorId;

    /**
     * Label or note from annotator 1.
     */
    @JsonProperty("annotator1")
    @CsvBindByName(column = "annotator1")
    @JsonPropertyDescription("Label/note provided by annotator 1.")
    private String annotator1;

    /**
     * Label or note from annotator 2.
     */
    @JsonProperty("annotator2")
    @CsvBindByName(column = "annotator2")
    @JsonPropertyDescription("Label/note provided by annotator 2.")
    private String annotator2;

    /**
     * Label or note from annotator 3.
     */
    @JsonProperty("annotator3")
    @CsvBindByName(column = "annotator3")
    @JsonPropertyDescription("Label/note provided by annotator 3.")
    private String annotator3;

    /**
     * Whether the audio/text was marked as inaudible.
     */
    @JsonProperty("inaudible")
    @CsvBindByName(column = "inaudible")
    @JsonPropertyDescription("Flag indicating whether the transcript was marked inaudible.")
    private boolean inaudible;

    /**
     * Whether multiple questions were detected.
     */
    @JsonProperty("multiple_questions")
    @CsvBindByName(column = "multiple_questions")
    @JsonPropertyDescription("True if the utterance bundles multiple questions.")
    private boolean multipleQuestions;

    /**
     * Whether the questions are affirmative in nature.
     */
    @JsonProperty("affirmative_questions")
    @CsvBindByName(column = "affirmative_questions")
    @JsonPropertyDescription("True if questions are affirmative in nature.")
    private boolean affirmativeQuestions;

    /**
     * Gold or annotated clarity label.
     */
    @JsonProperty("clarity_label")
    @CsvBindByName(column = "clarity_label")
    @JsonPropertyDescription("Ground-truth clarity taxonomy label.")
    private String clarityLabel;

    /**
     * Gold or annotated evasion label.
     */
    @JsonProperty("evasion_label")
    @CsvBindByName(column = "evasion_label")
    @JsonPropertyDescription("Ground-truth evasion taxonomy label.")
    private String evasionLabel;

    /**
     * Whether this item is part of a test set.
     */
    @JsonProperty("test")
    @JsonPropertyDescription("True if the QA belongs to the test split.")
    private boolean test;

    /**
     * Whether the item is considered valid.
     */
    @JsonProperty("valid")
    @JsonPropertyDescription("True if this QA is part of the validation split.")
    private boolean valid;

    /**
     * Whether the item is used for training.
     */
    @JsonProperty("train")
    @JsonPropertyDescription("True if this QA is part of the training split.")
    private boolean train;

    @JsonProperty("evaluation")
    @JsonPropertyDescription("True if this QA is part of the evaluation subset.")
    private boolean evaluation;

    /**
     * Embedding for the combined question and answer.
     */
    @JsonIgnore
    @CsvIgnore
    private double[] questionAnswerEmbedding;

    /**
     * Embedding for the question only.
     */
    @JsonIgnore
    @CsvIgnore
    private double[] questionEmbedding;

    /**
     * Embedding for the answer only.
     */
    @JsonIgnore
    @CsvIgnore
    private double[] answerEmbedding;
}