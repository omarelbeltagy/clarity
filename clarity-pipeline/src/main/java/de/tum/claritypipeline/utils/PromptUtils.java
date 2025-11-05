package de.tum.claritypipeline.utils;

import de.tum.clarityneo4j.model.Neo4jEmbeddingSearchResult;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.RaqProperties;
import de.tum.claritypipeline.model.config.ResponseFormat;
import de.tum.claritypipeline.model.core.Category;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.model.core.Taxonomy;
import de.tum.claritypipeline.service.EmbeddingService;
import de.tum.clarityutils.JsonScheme;
import jdk.jfr.Description;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class PromptUtils {
    private static final String PLACEHOLDER_QUESTION = "{question}";
    private static final String PLACEHOLDER_CONTEXT = "{context}";
    private static final String PLACEHOLDER_ONTOLOGY = "{ontology}";
    private static final String PLACEHOLDER_TAXONOMY = "{taxonomy}";
    private static final String PLACEHOLDER_RAQ_EXAMPLES = "{raq_examples}";
    private static final String PLACEHOLDER_CLASSIFICATION_RESULT = "{classification_result}";

    private static final String TEXT_FORMAT_SUFFIX = """
            
            ---
            
            Return only the label in the format "Label: <label>". No additional text or metadata.
            """;

    private static final String JSON_FORMAT_TEMPLATE = """
            
            ---
            
            Respond ONLY with valid minified JSON that strictly follows this schema. Do not include any text, comments, explanations, or markdown.
            %s
            """;

    @Description("Experimental. Not generic.")
    private static String injectExamplesWithRaq(
            String prompt, RaqProperties raqProperties, ClassificationRequest request) {
        if (prompt == null || raqProperties == null) {
            throw new IllegalArgumentException("Arguments must not be null");
        }
        if (!raqProperties.isEnabled()) {
            return prompt;
        }
        EmbeddingService service = EmbeddingService.getInstance();
        double[] requestEmbedding;
        switch (raqProperties.getEmbeddingIndex()) {
            case QA_ANSWER -> requestEmbedding = service.generateEmbeddings(request.getAnswer());
            case QA_QUESTION -> requestEmbedding = service.generateEmbeddings(request.getQuestion());
            case QA_QUESTION_AND_ANSWER ->
                    requestEmbedding = service.generateEmbeddings(request.getQuestion() + "\n" + request.getAnswer());
            default -> throw new IllegalArgumentException(
                    "Unsupported embedding index: " + raqProperties.getEmbeddingIndex());
        }
        List<Neo4jEmbeddingSearchResult<QA>> similarExamples = service.searchSimilar(
                raqProperties.getEmbeddingIndex().getIndexName(), requestEmbedding, 128,
                QA.class);

        similarExamples = similarExamples.stream()
                                         .collect(Collectors.groupingBy(
                                                 result -> result.getNode().getClarityLabel(),
                                                 LinkedHashMap::new,
                                                 Collectors.toList()
                                         ))
                                         .values().stream()
                                         .flatMap(list -> list.stream().limit(raqProperties.getK()))
                                         .toList();


        String examplesString = similarExamples.stream()
                                               .map(Neo4jEmbeddingSearchResult::getNode)
                                               .map(PromptUtils::buildExampleString)
                                               .collect(Collectors.joining("\n"));

        return prompt.replace(PLACEHOLDER_RAQ_EXAMPLES, examplesString);
    }

    private static String buildExampleString(QA qa) {
        StringBuilder sb = new StringBuilder();
        sb.append("Question: ").append(qa.getQuestion()).append("\n");
        sb.append("Answer: ").append(qa.getInterviewAnswer()).append("\n");
        sb.append("Label: ").append(qa.getClarityLabel()).append("\n");
        return sb.toString();
    }

    public static <T> String replacePrompt(
            ClassificationRequest classificationRequest,
            String prompt,
            ResponseFormat responseFormat,
            boolean injectResponseFormat,
            Taxonomy taxonomy,
            RaqProperties raqProperties,
            Class<T> resultClass
    ) {
        if (prompt == null || classificationRequest == null) {
            throw new IllegalArgumentException("Arguments must not be null");
        }

        String ontology = buildOntologyString(taxonomy.getCategories());
        prompt = replacePlaceholders(prompt, classificationRequest, ontology);
        if (injectResponseFormat) {
            prompt = appendResponseFormatInstructions(prompt, responseFormat, resultClass);
        }
        if (raqProperties.isEnabled()) {
            prompt = injectExamplesWithRaq(prompt, raqProperties, classificationRequest);
        }
        return prompt;
    }

    private static String buildOntologyString(List<Category> categories) {
        return IntStream.range(0, categories.size())
                        .mapToObj(i -> formatCategory(i + 1, categories.get(i)))
                        .collect(Collectors.joining());
    }

    private static String formatCategory(int index, Category category) {
        return String.format("%d. %s - %s", index, category.getName(), category.getDescription());
    }

    private static String replacePlaceholders(
            String prompt,
            ClassificationRequest request,
            String ontology
    ) {
        return prompt
                .replace(PLACEHOLDER_QUESTION, request.getQuestion())
                .replace(PLACEHOLDER_CONTEXT, request.getContext())
                .replace(PLACEHOLDER_ONTOLOGY, ontology)
                .replace(PLACEHOLDER_TAXONOMY, ontology);
    }

    private static <T> String appendResponseFormatInstructions(
            String prompt, ResponseFormat format, Class<T> resultClass) {
        return switch (format) {
            case TEXT -> prompt + TEXT_FORMAT_SUFFIX;
            case JSON_OBJECT -> prompt + JSON_FORMAT_TEMPLATE.formatted(getJsonSchema(resultClass));
        };
    }

    private static <T> String getJsonSchema(Class<T> resultClass) {
        return new JsonScheme<>(resultClass).getPropertiesString();
    }

    public static <T> String replaceJudgementPrompt(
            ClassificationRequest request, ClassificationResult initialResult, String prompt,
            ResponseFormat responseFormat, boolean injectResponseFormat, Taxonomy taxonomy, RaqProperties raqProperties,
            Class<T> resultClass
    ) {
        prompt = replacePrompt(request, prompt, responseFormat, injectResponseFormat, taxonomy, raqProperties,
                               resultClass);
        String classificationResultStr = buildClassificationResult(initialResult);
        prompt = prompt.replace(PLACEHOLDER_CLASSIFICATION_RESULT, classificationResultStr);
        return prompt;
    }

    private static String buildClassificationResult(ClassificationResult result) {
        StringBuilder sb = new StringBuilder();
        sb.append("Name: ").append(result.getName()).append("\n");
        if (result.getExplanation() != null && !result.getExplanation().isEmpty()) {
            sb.append("Explanation: ").append(result.getExplanation()).append("\n");
        }
        return sb.toString();
    }
}
