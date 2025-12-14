package de.tum.claritypipeline.utils;

import de.tum.clarityneo4j.model.Neo4jEmbeddingSearchResult;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.config.GlobalConfig;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.model.core.Taxonomy;

import java.lang.reflect.Field;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class PromptUtils {
    private static final String PLACEHOLDER_CONTEXT = "{context}";
    private static final String PLACEHOLDER_ONTOLOGY = "{ontology}";
    private static final String PLACEHOLDER_TAXONOMY = "{taxonomy}";
    private static final String PLACEHOLDER_EXAMPLES = "{examples}";
    private static final String PLACEHOLDER_RESPONSE_FORMAT = "{response_format}";

    private static String buildRaqExampleString(QA qa, Taxonomy taxonomy) {
        try {
            Field field = qa.getClass().getDeclaredField(
                    taxonomy.getLabelProperty());
            field.setAccessible(true);
            Object value = field.get(qa);
            String label = value.toString();
            StringBuilder sb = new StringBuilder();
            sb.append("Question: ").append(qa.getQuestion()).append("\n");
            sb.append("Answer: ").append(qa.getInterviewAnswer()).append("\n");
            sb.append("Label: ").append(label).append("\n");
            return sb.toString();
        } catch (NoSuchFieldException |
                 IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    public static String replacePrompt(
            ClassificationRequest classificationRequest,
            ModelProperties modelProperties,
            String jsonScheme
    ) {
        if (modelProperties == null || classificationRequest == null) {
            throw new IllegalArgumentException("Arguments must not be null.");
        }
        if (modelProperties.getPrompt() == null || modelProperties.getPrompt().isEmpty()) {
            throw new IllegalArgumentException("Prompt is not set.");
        }
        if (classificationRequest.getTaxonomy() == null || classificationRequest.getTaxonomy().getCategories()
                                                                                .isEmpty()) {
            throw new IllegalArgumentException("Taxonomy is empty.");
        }

        return replacePlaceholders(modelProperties.getPrompt(), modelProperties,
                                   classificationRequest, jsonScheme);
    }

    private static String buildOntologyString(List<Taxonomy.Category> categories) {
        return IntStream.range(0, categories.size())
                        .mapToObj(i -> formatCategory(i + 1, categories.get(i)))
                        .collect(Collectors.joining());
    }

    private static String buildExamplesString(List<Taxonomy.Category> categories) {
        StringBuilder sb = new StringBuilder();
        for (Taxonomy.Category category : categories) {
            if (category.getExamples() != null) {
                for (Taxonomy.Category.TaxonomyExample example : category.getExamples()) {
                    sb.append("Question: ").append(example.getQuestion()).append("\n");
                    sb.append("Answer: ").append(example.getAnswer()).append("\n");
                    sb.append("Label: ").append(category.getName()).append("\n");
                    sb.append("Explanation: ").append(example.getExplanation()).append("\n\n");
                }
            }
        }
        return sb.toString();
    }

    private static String formatCategory(int index, Taxonomy.Category category) {
        return String.format("%d. %s - %s", index, category.getName(), category.getDescription());
    }

    private static String replacePlaceholders(
            String prompt,
            ModelProperties modelProperties,
            ClassificationRequest request,
            String jsonScheme
    ) {
        prompt = prompt
                .replace(PLACEHOLDER_CONTEXT,
                         buildContext(request.getQa().getInterviewQuestion(), request.getQa().getInterviewAnswer()))
                .replace(PLACEHOLDER_ONTOLOGY, buildOntologyString(request.getTaxonomy().getCategories()))
                .replace(PLACEHOLDER_TAXONOMY, buildOntologyString(request.getTaxonomy().getCategories()))
                .replace(PLACEHOLDER_RESPONSE_FORMAT,
                         getResponseFormatInstructions(modelProperties.getResponseFormat(), jsonScheme));
        if (modelProperties.getRagProperties() != null && modelProperties.getRagProperties().isEnabled()) {
            prompt = prompt
                    .replace(PLACEHOLDER_EXAMPLES,
                             buildRagExamples(prompt, modelProperties.getRagProperties(), request));
        } else {
            prompt = prompt
                    .replace(PLACEHOLDER_EXAMPLES, buildExamplesString(request.getTaxonomy().getCategories()));
        }
        prompt = replacePlaceholdersDynamic(prompt, request.getQa());
        return prompt;
    }

    private static String buildRagExamples(
            String prompt, ModelProperties.RagProperties ragProperties, ClassificationRequest request) {
        if (prompt == null || ragProperties == null) {
            throw new IllegalArgumentException("Arguments must not be null");
        }
        if (!ragProperties.isEnabled()) {
            return prompt;
        }
        double[] requestEmbedding = switch (ragProperties.getEmbeddingIndex()) {
            case QA_ANSWER -> request.getQa().getAnswerEmbedding();
            case QA_QUESTION -> request.getQa().getQuestionEmbedding();
            case QA_QUESTION_AND_ANSWER -> request.getQa().getQuestionAnswerEmbedding();
        };
        if (requestEmbedding == null || requestEmbedding.length == 0) {
            throw new IllegalArgumentException("Embedding must not be null or empty.");
        }
        List<Neo4jEmbeddingSearchResult<QA>> similarExamples = GlobalConfig.NEO4J_CLIENT.similaritySearch(
                                                                                   ragProperties.getEmbeddingIndex().getIndexName(), requestEmbedding, 256, QA.class).stream()
                                                                                        .filter(example -> !example.getNode()
                                                                                                                   .isTest())
                                                                                        .toList();

        similarExamples = similarExamples.stream()
                                         .collect(Collectors.groupingBy(
                                                 r -> {
                                                     try {
                                                         Field field = r.getNode().getClass().getDeclaredField(
                                                                 request.getTaxonomy().getLabelProperty());
                                                         field.setAccessible(true);
                                                         Object value = field.get(r.getNode());
                                                         return value.toString();
                                                     } catch (NoSuchFieldException |
                                                              IllegalAccessException e) {
                                                         throw new RuntimeException(e);
                                                     }
                                                 },
                                                 LinkedHashMap::new,
                                                 Collectors.toList()
                                         ))
                                         .values().stream()
                                         .flatMap(list -> list.stream()
                                                              .limit(ragProperties.getK()))
                                         .filter(Objects::nonNull)
                                         .toList();


        return similarExamples.stream()
                              .map(Neo4jEmbeddingSearchResult::getNode)
                              .map(qa -> buildRaqExampleString(qa, request.getTaxonomy()))
                              .collect(Collectors.joining("\n"));
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

    public static String replacePlaceholdersDynamic(String text, QA qa) {
        Pattern p = Pattern.compile("\\{([^}]+)}");
        Matcher m = p.matcher(text);

        StringBuilder out = new StringBuilder();

        while (m.find()) {
            String name = m.group(1);
            Object value = getFieldValue(qa, name);
            String replacement = value != null ? value.toString() : m.group(0);
            m.appendReplacement(out, Matcher.quoteReplacement(replacement));
        }

        m.appendTail(out);
        return out.toString();
    }

    private static Object getFieldValue(Object obj, String fieldName) {
        try {
            Field f = obj.getClass().getDeclaredField(fieldName);
            f.setAccessible(true);
            return f.get(obj);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            return null;
        }
    }


    private static String getResponseFormatInstructions(ModelProperties.ResponseFormat format, String jsonScheme) {
        return switch (format) {
            case TEXT -> "Return only the label in the format \"Label: <label>\". No additional text or metadata.";
            case JSON_OBJECT -> "Respond with a JSON object in the following format:\n%s".formatted(jsonScheme);
        };
    }
}
