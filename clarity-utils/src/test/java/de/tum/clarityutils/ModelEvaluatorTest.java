package de.tum.clarityutils;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;

import java.util.Arrays;
import java.util.List;

public class ModelEvaluatorTest {
    private final Logger log = org.slf4j.LoggerFactory.getLogger(ModelEvaluatorTest.class);

    private final List<String> labels = Arrays.asList("Clear Reply", "Ambivalent", "Clear Non-Reply");
    private final List<String> predictions = Arrays.asList("Clear Reply", "Ambivalent", "Clear Reply",
                                                           "Clear Non-Reply", "Clear Reply", "Clear Reply",
                                                           "Ambivalent", "Clear Reply", "Clear Reply",
                                                           "Clear Non-Reply", "Clear Reply", "Clear Reply");
    private final List<String> expected = Arrays.asList("Clear Reply", "Clear Reply", "Clear Reply", "Clear Non-Reply",
                                                        "Clear Reply", "Clear Reply", "Ambivalent", "Clear Reply",
                                                        "Clear Non-Reply", "Clear Reply", "Clear Reply",
                                                        "Ambivalent");

    private final ModelEvaluator eval = new ModelEvaluator(labels, predictions, expected);

    @Test
    public void testAccuracy() {
        double accuracy = eval.accuracy();
        log.info("Accuracy: {}", accuracy);
    }

    @Test
    public void testRecall() {
        double recall = eval.recall();
        log.info("Recall: {}", recall);
    }

    @Test
    public void testPrecision() {
        double precision = eval.precision();
        log.info("Precision: {}", precision);
    }

    @Test
    public void testMicroF1() {
        double microF1 = eval.microF1();
        log.info("Micro F1: {}", microF1);
    }

    @Test
    public void testMacroF1() {
        double macroF1 = eval.macroF1();
        log.info("Macro F1: {}", macroF1);
    }
}
