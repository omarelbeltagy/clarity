package de.tum.claritypipeline.utils;

import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.model.core.Taxonomy;

public class PipelineUtils {

    /**
     * Build a ClassificationRequest from a QA object.
     *
     * @param qa The QA object.
     * @return The constructed ClassificationRequest.
     */
    public static ClassificationRequest buildRequest(QA qa, Taxonomy taxonomy) {
        return ClassificationRequest.builder()
                                    .qa(qa)
                                    .question(qa.getQuestion())
                                    .context(buildContext(qa.getInterviewQuestion(),
                                                          qa.getInterviewAnswer()))
                                    .taxonomy(taxonomy)
                                    .answer(qa.getInterviewAnswer())
                                    .build();
    }

    private static String buildContext(String interviewQuestion, String interviewAnswer) {
        StringBuilder contextBuilder = new StringBuilder();
        if (interviewQuestion.startsWith("Q. ")) {
            interviewQuestion = interviewQuestion.substring(3);
        }
        contextBuilder.append("Interviewer: ").append(interviewQuestion).append("\n");
        contextBuilder.append("Answer: ").append(interviewAnswer).append("\n");
        return contextBuilder.toString();
    }
}
