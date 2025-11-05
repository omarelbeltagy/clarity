package de.tum.claritypipeline.utils;

import de.tum.clarityneo4j.model.Neo4jEmbeddingSearchResult;
import de.tum.claritypipeline.model.*;
import de.tum.claritypipeline.model.properties.RaqProperties;
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

    private static final String TEXT_FORMAT_SUFFIX = """
            
            ---
            
            Return only the label in the format "Label: <label>". No additional text or metadata.
            """;

    private static final String JSON_FORMAT_TEMPLATE = """
            
            ---
            
            Answer strictly in the following JSON format:
            %s
            """;

    @Description("Experimental. Not generic.")
    public static String injectExamplesWithRaq(
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

    public static String replacePrompt(
            ClassificationRequest classificationRequest,
            String prompt,
            ResponseFormat responseFormat,
            boolean injectResponseFormat,
            Taxonomy taxonomy
    ) {
        if (prompt == null || classificationRequest == null) {
            throw new IllegalArgumentException("Arguments must not be null");
        }

        String ontology = buildOntologyString(taxonomy.getCategories());
        String processedPrompt = replacePlaceholders(prompt, classificationRequest, ontology);
        if (injectResponseFormat) {
            return appendResponseFormatInstructions(processedPrompt, responseFormat);
        } else {
            return processedPrompt;
        }
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

    private static String appendResponseFormatInstructions(String prompt, ResponseFormat format) {
        return switch (format) {
            case TEXT -> prompt + TEXT_FORMAT_SUFFIX;
            case JSON_OBJECT -> prompt + JSON_FORMAT_TEMPLATE.formatted(getJsonSchema());
        };
    }

    private static String getJsonSchema() {
        return new JsonScheme<>(ClassificationResult.class).getPropertiesString();
    }
}
