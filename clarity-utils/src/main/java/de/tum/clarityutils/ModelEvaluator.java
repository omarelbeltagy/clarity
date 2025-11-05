package de.tum.clarityutils;

import org.nd4j.evaluation.EvaluationAveraging;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Evaluates classification predictions against expected labels using ND4J's Evaluation.
 *
 * <p>This class accepts immutable lists of label names, predicted label names and expected
 * label names. The label list defines the one-hot encoding index for each label. Metrics
 * (accuracy, recall, precision and F1 scores) are computed lazily on first request and
 * cached for subsequent calls.</p>
 */
public class ModelEvaluator {
    private final Evaluation eval;
    private final List<String> labels;
    private final List<String> predictions;
    private final List<String> expected;

    private boolean initialized = false;

    /**
     * Create a ModelEvaluator.
     *
     * @param labels      list of label names; the order defines indices for one-hot encoding
     * @param predictions predicted label names; must be same size as expected
     * @param expected    expected label names; must be same size as predictions
     * @throws IllegalArgumentException if predictions and expected sizes differ
     */
    public ModelEvaluator(List<String> labels, List<String> predictions, List<String> expected) {
        if (predictions.size() != expected.size()) {
            throw new IllegalArgumentException("predictions and expected lists must have the same size.");
        }
        this.labels = List.copyOf(labels);
        this.predictions = List.copyOf(predictions);
        this.expected = List.copyOf(expected);
        this.eval = new Evaluation(this.labels);
    }

    /**
     * Returns overall accuracy computed from the provided predictions and expected labels.
     *
     * @return accuracy in range [0.0, 1.0]
     */
    public double accuracy() {return withEval(eval::accuracy);}

    /**
     * Returns the macro-averaged recall computed from predictions and expected labels.
     *
     * @return recall in range [0.0, 1.0]
     */
    public double recall() {return withEval(eval::recall);}

    /**
     * Returns the macro-averaged precision computed from predictions and expected labels.
     *
     * @return precision in range [0.0, 1.0]
     */
    public double precision() {return withEval(eval::precision);}

    /**
     * Returns the micro-averaged F1 score.
     *
     * @return micro F1 in range [0.0, 1.0]
     */
    public double microF1() {return withEval(() -> eval.f1(EvaluationAveraging.Micro));}

    /**
     * Returns the macro-averaged F1 score.
     *
     * @return macro F1 in range [0.0, 1.0]
     */
    public double macroF1() {return withEval(() -> eval.f1(EvaluationAveraging.Macro));}

    /**
     * Returns the weighted-averaged F1 score.
     *
     * @return weighted F1 in range [0.0, 1.0]
     */
    private double withEval(SupplierDouble metric) {
        initEval();
        return metric.get();
    }

    /**
     * Initializes the Evaluation object by feeding in all predictions and expected labels.
     * This method is called lazily on first metric request.
     */
    private void initEval() {
        if (initialized) return;

        Map<String, Integer> labelToIndex = IntStream.range(0, labels.size())
                                                     .collect(HashMap::new, (m, i) -> m.put(labels.get(i), i),
                                                              Map::putAll);

        IntStream.range(0, predictions.size()).forEach(i -> {
            int predIdx = labelToIndex.get(predictions.get(i));
            int actIdx = labelToIndex.get(expected.get(i));

            INDArray predicted = oneHot(predIdx);
            INDArray actual = oneHot(actIdx);

            eval.eval(actual, predicted);
        });

        initialized = true;
    }

    /**
     * Creates a one-hot encoded INDArray for the given label index.
     *
     * @param index index of the label to set to 1
     * @return one-hot encoded INDArray
     */
    private INDArray oneHot(int index) {
        INDArray arr = Nd4j.zeros(1, labels.size());
        arr.putScalar(0, index, 1.0);
        return arr;
    }

    /**
     * Functional interface for supplying double values.
     */
    @FunctionalInterface
    private interface SupplierDouble {
        double get();
    }
}
