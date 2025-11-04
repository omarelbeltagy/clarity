package de.tum.claritypipeline.utils;

import de.tum.claritypipeline.model.*;
import de.tum.clarityutils.JsonScheme;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class PromptUtils {
    private static final String PLACEHOLDER_QUESTION = "{question}";
    private static final String PLACEHOLDER_CONTEXT = "{context}";
    private static final String PLACEHOLDER_ONTOLOGY = "{ontology}";
    private static final String PLACEHOLDER_TAXONOMY = "{taxonomy}";

    private static final String TEXT_FORMAT_SUFFIX = """
            
            ---
            
            Return only the label in the format "Label: <label>". No additional text or metadata.
            """;

    private static final String JSON_FORMAT_TEMPLATE = """
            
            ---
            
            Answer strictly in the following JSON format:
            %s
            """;

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
