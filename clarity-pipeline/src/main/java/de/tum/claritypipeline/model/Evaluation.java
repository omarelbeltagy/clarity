package de.tum.claritypipeline.model;

import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import lombok.*;

/**
 * Holds evaluation metrics for a classification experiment.
 *
 * <p>Metrics are typical classification measures such as accuracy, precision, recall and F1 scores.
 */
@Node(label = "Evaluation")
@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class Evaluation extends Neo4jNode {

    /**
     * Overall accuracy of the classifier (correct / total).
     */
    private double accuracy;

    /**
     * Precision metric (positive predictive value).
     */
    private double precision;

    /**
     * Recall metric (sensitivity).
     */
    private double recall;

    /**
     * Macro-averaged F1 score across classes.
     */
    private double macroF1;

    /**
     * Macro-averaged F1 score across classes, rounded to 2 decimal places.
     */
    private double macroF1Rounded;

    /**
     * Micro-averaged F1 score across classes.
     */
    private double microF1;
}
