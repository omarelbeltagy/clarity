package de.tum.claritypipeline.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.classification.FailureModesResult;
import de.tum.claritypipeline.model.classification.PatchResult;
import de.tum.claritypipeline.model.config.GlobalConfig;
import de.tum.claritypipeline.model.config.PromptEnhancingProperties;
import de.tum.claritypipeline.model.core.PromptEnhancingIteration;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.model.core.Taxonomy;
import de.tum.claritypipeline.model.relation.*;
import de.tum.claritypipeline.utils.PromptUtils;
import de.tum.clarityutils.SerializationUtils;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class PromptEnhancer {
    private final static String PLACEHOLDER_PROMPT = "{dump_prompt}";
    private final static String PLACEHOLDER_FAILURE_TRACES = "{dump_failure_traces}";
    private final static String PLACEHOLDER_FAILURE_MODE_ANALYSIS = "{dump_failure_mode_analysis}";
    /**
     * Logger instance for logging information and errors.
     */
    private final Logger log = org.slf4j.LoggerFactory.getLogger(PromptEnhancer.class);
    /**
     * Neo4j client for database operations.
     */
    private final Neo4jClient client;

    private final PromptEnhancingProperties properties;

    public PromptEnhancer(String propertiesFilePath) throws IOException {
        this.properties = PromptEnhancingProperties.load(propertiesFilePath);
        this.client = GlobalConfig.NEO4J_CLIENT;
    }

    private void outputPromptToFile(String prompt) {
        if (properties.getOutputPrompt() == null || properties.getOutputPrompt().isBlank()) {
            return;
        }
        try {
            if (properties.getOutputPrompt().endsWith(".yaml") || properties.getOutputPrompt().endsWith(".yml")) {
                StringBuilder yamlContent = new StringBuilder("prompt: |\n");
                for (String line : prompt.split("\n")) {
                    yamlContent.append("  ").append(line).append("\n");
                }
                SerializationUtils.writeStringToFile(properties.getOutputPrompt(), yamlContent.toString());
            } else {
                SerializationUtils.writeStringToFile(properties.getOutputPrompt(), prompt);
            }
        } catch (Exception e) {
            log.error("Failed to write prompt to file: {}", e.getMessage(), e);
        }
    }

    private void outputTaxonomyToFile(Taxonomy taxonomy) {
        if (properties.getOutputTaxonomy() == null || properties.getOutputTaxonomy().isBlank()) {
            return;
        }
        try {
            ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());
            yamlMapper.findAndRegisterModules();
            yamlMapper.writerWithDefaultPrettyPrinter();
            yamlMapper.writeValue(
                    new File(properties.getOutputTaxonomy()),
                    taxonomy
            );
        } catch (IOException e) {
            log.error("Failed to write taxonomy to file: {}", e.getMessage(), e);
        }
    }

    private ClassificationResult classifySingle(
            String prompt, ClassificationRequest classificationRequest
    ) {
        for (int i = 0; i < properties.getAttempts(); i++) {
            if (i > 0) {
                log.info("Retrying classification attempt {} for question: {}",
                         i + 1, classificationRequest.getQuestion());
                try {
                    Thread.sleep(i * 1000L);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
            }
            try {
                prompt = PromptUtils.replacePlaceholders(prompt, properties.getModel(), classificationRequest,
                                                         ClassificationResult.JSON_SCHEME);
                ClassificationResult classificationResult = properties.getModel().getClient()
                                                                      .makeRequest(prompt, ClassificationResult.class);
                if (classificationResult != null) {
                    return classificationResult;
                }
                log.warn("Classification attempt {} for question: {} returned null response.",
                         i + 1, classificationRequest.getQuestion());
            } catch (Exception e) {
                log.warn("Classification attempt {} failed: {}", i + 1, e.getMessage());
            }
        }
        log.error("All classification attempts failed for question: {}",
                  classificationRequest.getQuestion());
        throw new RuntimeException("All classification attempts failed.");
    }

    private String getDiagnosePrompt(Taxonomy taxonomy) {
        ClassificationRequest dummyRequest = ClassificationRequest.builder()
                                                                  .qa(new QA())
                                                                  .question("question")
                                                                  .context("context")
                                                                  .taxonomy(taxonomy)
                                                                  .answer("answer")
                                                                  .build();

        return PromptUtils.replacePlaceholders(properties.getEnhancementPromptDiagnose(),
                                               properties.getModel(),
                                               dummyRequest,
                                               FailureModesResult.JSON_SCHEME);
    }

    private String getPatchPrompt(Taxonomy taxonomy) {
        ClassificationRequest dummyRequest = ClassificationRequest.builder()
                                                                  .qa(new QA())
                                                                  .question("question")
                                                                  .context("context")
                                                                  .taxonomy(taxonomy)
                                                                  .answer("answer")
                                                                  .build();

        return PromptUtils.replacePlaceholders(properties.getEnhancementPromptPatch(),
                                               properties.getModel(),
                                               dummyRequest,
                                               PatchResult.JSON_SCHEME);
    }

    public void enhance() {
        List<QA> qas = fetchQAs();
        Collections.shuffle(qas);

        String currentClassificationPrompt = properties.getClassificationPrompt();
        Taxonomy currentTaxonomy = properties.getTaxonomy();
        for (int i = 0; i < properties.getIterations(); i++) {
            PromptEnhancingIteration iteration = PromptEnhancingIteration.builder()
                                                                         .iterationNumber(i + 1)
                                                                         .initialPrompt(currentClassificationPrompt)
                                                                         .build();
            saveIteration(iteration);

            List<ClassificationTask> tasks = classifyQAsForIteration(qas, currentTaxonomy,
                                                                     currentClassificationPrompt, i);
            if (tasks.isEmpty()) {
                log.info("No more QAs to enhance. Stopping enhancement process.");
                break;
            }
            batchSaveClassifications(tasks, iteration);

            List<ClassificationTask> failedTasks = filterFailedClassifications(tasks);
            log.info("Iteration {}: {} out of {} QAs misclassified.",
                     i + 1, failedTasks.size(), tasks.size());
            if (failedTasks.isEmpty()) {
                log.info("All QAs classified correctly. No prompt enhancement needed.");
            } else {
                String failureTraces = buildFailureTraces(failedTasks);

                String diagnosePrompt = getDiagnosePrompt(currentTaxonomy)
                        .replace(PLACEHOLDER_FAILURE_TRACES, failureTraces)
                        .replace(PLACEHOLDER_PROMPT, currentClassificationPrompt);

                iteration.setDiagnoseRequest(diagnosePrompt);

                FailureModesResult failureModesResult =
                        properties.getModel().getClient()
                                  .makeRequest(diagnosePrompt, FailureModesResult.class);

                log.info("Identified {} failure modes during diagnosis.",
                         failureModesResult.getFailureModes() != null
                                 ? failureModesResult.getFailureModes().size() : 0);

                iteration.setDiagnoseResult(SerializationUtils.serialize(failureModesResult));

                String patchPrompt = getPatchPrompt(currentTaxonomy)
                        .replace(PLACEHOLDER_FAILURE_MODE_ANALYSIS, buildFailureModes(failureModesResult))
                        .replace(PLACEHOLDER_PROMPT, currentClassificationPrompt);

                iteration.setPatchRequest(patchPrompt);

                PatchResult patchResult =
                        properties.getModel().getClient()
                                  .makeRequest(patchPrompt, PatchResult.class);

                if (patchResult != null) {
                    iteration.setPatchResult(SerializationUtils.serialize(patchResult));
                    if (patchResult.getRevisedPrompt() != null && !patchResult.getRevisedPrompt().isBlank()) {
                        currentClassificationPrompt = patchResult.getRevisedPrompt();
                    } else {
                        log.warn("Revised prompt in patch result was null or empty, retaining current prompt.");
                    }
                    if (patchResult.getRevisedTaxonomy() != null) {
                        currentTaxonomy = Taxonomy.builder()
                                                  .name(currentTaxonomy.getName())
                                                  .description(currentTaxonomy.getDescription())
                                                  .version(currentTaxonomy.getVersion())
                                                  .labelProperty(currentTaxonomy.getLabelProperty())
                                                  .categories(patchResult.getRevisedTaxonomy())
                                                  .build();
                        validateNewTaxonomy(currentTaxonomy);
                        iteration.setRevisedTaxonomy(SerializationUtils.serialize(currentTaxonomy));
                    } else {
                        log.warn("Revised taxonomy in patch result was null, retaining current taxonomy.");
                    }
                    iteration.setRevisedPrompt(currentClassificationPrompt);
                    client.updateNode(iteration);
                } else {
                    log.warn("Patch result was null or empty, retaining current prompt.");
                }
            }

            log.info("Enhancement iteration {} completed with {} QAs.", i + 1, tasks.size());
        }
        outputPromptToFile(currentClassificationPrompt);
        outputTaxonomyToFile(currentTaxonomy);

        log.info("Prompt enhancement process completed.");
    }

    private void validateNewTaxonomy(Taxonomy taxonomy) {
        Set<String> originalLabels = properties.getTaxonomy().getCategories()
                                               .stream()
                                               .map(Taxonomy.Category::getName)
                                               .collect(Collectors.toSet());
        for (Taxonomy.Category category : taxonomy.getCategories()) {
            if (!originalLabels.contains(category.getName())) {
                throw new IllegalArgumentException(
                        "New categories cannot be added to the taxonomy during prompt enhancement. "
                                + "Offending category: " + category.getName());
            }
        }
        List<Taxonomy.Category> completeCategories = new ArrayList<>(taxonomy.getCategories());
        for (Taxonomy.Category originalCategory : properties.getTaxonomy().getCategories()) {
            boolean exists = taxonomy.getCategories().stream()
                                     .anyMatch(cat -> cat.getName().equals(originalCategory.getName()));
            if (!exists) {
                completeCategories.add(originalCategory);
            }
        }
        taxonomy.setCategories(completeCategories);
    }

    private List<ClassificationTask> filterFailedClassifications(List<ClassificationTask> tasks) {
        return tasks.stream()
                    .filter(task -> task.expectedCategory != null
                            && !task.expectedCategory.equalsIgnoreCase(task.category.getName()))
                    .toList();
    }

    private String buildFailureModes(FailureModesResult failureModesResult) {
        StringBuilder sb = new StringBuilder();
        if (failureModesResult.getFailureModes() != null) {
            for (FailureModesResult.FailureMode mode : failureModesResult.getFailureModes()) {
                sb.append("Failure Mode: ").append(mode.getName()).append("\n");
                sb.append("Description: ").append(mode.getDescription()).append("\n");
                sb.append("Prompt Drivers:\n");
                if (mode.getPromptDrivers() != null) {
                    for (FailureModesResult.FailureMode.PromptDriver driver : mode.getPromptDrivers()) {
                        sb.append("- ").append(driver.getExactOrParaphrasedLine())
                          .append(": ").append(driver.getWhyItMatters()).append("\n");
                    }
                }
                sb.append("----\n");
            }
        }
        return sb.toString();
    }

    private String buildFailureTraces(List<ClassificationTask> tasks) {
        StringBuilder sb = new StringBuilder();
        for (ClassificationTask task : tasks) {
            if (task.expectedCategory != null && !task.expectedCategory.equals(task.category.getName())) {
                sb.append("Question: ").append(task.qa.getQuestion()).append("\n");
                sb.append("Answer: ").append(task.qa.getInterviewAnswer()).append("\n");
                sb.append("Assigned Category: ").append(task.category.getName()).append("\n");
                sb.append("Expected Category: ").append(task.expectedCategory).append("\n");
                sb.append("Model Explanation: ").append(task.result.getExplanation()).append("\n");
                sb.append("----\n");
            }
        }
        return sb.toString();
    }

    private void saveIteration(PromptEnhancingIteration iteration) {
        client.saveNode(iteration);
        client.createRelation(createRelationObject(new HasIteration(),
                                                   properties.getElementId(), iteration.getElementId()));
    }

    private void batchSaveClassifications(List<ClassificationTask> tasks, PromptEnhancingIteration iteration) {
        client.batchSaveNodes(tasks.stream().map(t -> t.result).toList());

        List<Neo4jRelation> relations = new ArrayList<>();
        for (ClassificationTask task : tasks) {
            relations.add(createRelationObject(new HasClassification(),
                                               task.qa.getElementId(), task.result.getElementId()));
            if (task.category != null && task.category.getElementId() != null) {
                relations.add(createRelationObject(new BelongsTo(),
                                                   task.result.getElementId(), task.category.getElementId()));
            }
            relations.add(createRelationObject(new GeneratedBy(),
                                               task.result.getElementId(), properties.getElementId()));
            relations.add(createRelationObject(new BelongsToIteration(),
                                               task.result.getElementId(), iteration.getElementId()));
        }

        client.batchCreateRelations(relations);
    }

    /**
     * Create a relation object with the specified start and end node IDs.
     *
     * @param relation    The relation object to set the IDs on.
     * @param startNodeId The start node ID.
     * @param endNodeId   The end node ID.
     * @param <T>         The type of the relation.
     * @return The relation object with the IDs set.
     */
    private <T extends Neo4jRelation> T createRelationObject(T relation, String startNodeId, String endNodeId) {
        relation.setStartNodeId(startNodeId);
        relation.setEndNodeId(endNodeId);
        return relation;
    }

    private List<ClassificationTask> classifyQAsForIteration(
            List<QA> unusedQAs, Taxonomy taxonomy, String prompt, int i) {
        List<QA> selectedQAs = unusedQAs.stream()
                                        .limit(properties.getN())
                                        .toList();
        if (selectedQAs.isEmpty()) {
            log.info("No more unused QAs available for enhancement.");
            return Collections.emptyList();
        }
        unusedQAs.removeAll(selectedQAs);

        log.info("Starting classification for {} ({}) of {} QAs using {} threads for iteration {}",
                 properties.getName(),
                 properties.getVersion(),
                 selectedQAs.size(), properties.getWorkerThreads(), i + 1);

        List<ClassificationTask> tasks;
        AtomicInteger counter = new AtomicInteger();
        try (ExecutorService executor = Executors.newFixedThreadPool(properties.getWorkerThreads())) {
            List<CompletableFuture<ClassificationTask>> futures =
                    selectedQAs.stream()
                               .map(qa -> CompletableFuture.supplyAsync(() -> {
                                   try {
                                       ClassificationRequest request = buildRequest(qa, taxonomy);
                                       ClassificationResult result = classifySingle(prompt, request);
                                       Taxonomy.Category category = findAssignedCategory(result.getName());
                                       if (category != null) {
                                           result.setName(category.getName());
                                       }
                                       String expectedCategoryName = findExpectedCategory(qa);
                                       log.info("Classified QA as {} ({}/{})",
                                                result.getName() != null ? result.getName() : "UNKNOWN",
                                                counter.incrementAndGet(),
                                                selectedQAs.size());
                                       return new ClassificationTask(qa, result, category, expectedCategoryName);
                                   } catch (Exception e) {
                                       log.error("Error classifying QA {}: {}", qa.getElementId(), e.getMessage(),
                                                 e);
                                       return null;
                                   }
                               }, executor))
                               .toList();

            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();

            tasks = futures.stream()
                           .map(CompletableFuture::join)
                           .filter(task -> task != null && task.category != null)
                           .toList();
        }
        return tasks;
    }

    private ClassificationRequest buildRequest(QA qa, Taxonomy taxonomy) {
        return ClassificationRequest.builder()
                                    .qa(qa)
                                    .question(qa.getQuestion())
                                    .context(buildContext(qa.getInterviewQuestion(),
                                                          qa.getInterviewAnswer()))
                                    .taxonomy(taxonomy)
                                    .answer(qa.getInterviewAnswer())
                                    .build();
    }

    private String buildContext(String interviewQuestion, String interviewAnswer) {
        StringBuilder contextBuilder = new StringBuilder();
        if (interviewQuestion.startsWith("Q. ")) {
            interviewQuestion = interviewQuestion.substring(3);
        }
        contextBuilder.append("Interviewer: ").append(interviewQuestion).append("\n");
        contextBuilder.append("Answer: ").append(interviewAnswer).append("\n");
        return contextBuilder.toString();
    }

    private List<QA> fetchQAs() {
        return client.executeQuery(properties.getQuery(), QA.class);
    }

    private List<QA> filterNotUsedQAs(List<QA> qas) {
        String qaIds = qas.stream()
                          .map(qa -> "'" + qa.getElementId() + "'")
                          .collect(Collectors.joining(","));

        String query = String.format("""
                                             MATCH (n:%s)-[:%s]->(:%s)-[:%s]->(pep:%s)
                                             WHERE elementId(n) IN [%s]
                                             AND elementId(pep) = '%s'
                                             RETURN n
                                             """,
                                     Neo4jNode.getLabel(QA.class),
                                     Neo4jRelation.getType(HasClassification.class),
                                     Neo4jNode.getLabel(ClassificationResult.class),
                                     Neo4jRelation.getType(GeneratedBy.class),
                                     Neo4jNode.getLabel(PromptEnhancingProperties.class),
                                     qaIds,
                                     properties.getElementId()
        );

        Set<String> used = client.executeQuery(query, QA.class)
                                 .stream().map(QA::getElementId).collect(Collectors.toSet());

        return qas.stream()
                  .filter(qa -> !used.contains(qa.getElementId()))
                  .collect(Collectors.toCollection(ArrayList::new));
    }

    private Taxonomy.Category findAssignedCategory(String name) {
        String normalizedName = name.replaceAll("[ _-]", "");
        for (Taxonomy.Category category : properties.getTaxonomy().getCategories()) {
            String normalizedCategoryName = category.getName().replaceAll("[ _-]", "");
            if (normalizedCategoryName.equals(normalizedName)) {
                return category;
            }
        }
        for (Taxonomy.Category category : properties.getTaxonomy().getCategories()) {
            String normalizedCategoryName = category.getName().replaceAll("[ _-]", "");
            if (normalizedCategoryName.contains(normalizedName)
                    || normalizedName.contains(normalizedCategoryName)) {
                return category;
            }
        }
        return null;
    }

    private String findExpectedCategory(QA qa) {
        String propertyLabel = properties.getTaxonomy().getLabelProperty();
        try {
            Field field = qa.getClass().getDeclaredField(propertyLabel);
            field.setAccessible(true);
            Object value = field.get(qa);
            if (value == null) {
                return null;
            }
            return value.toString();
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    private record ClassificationTask(QA qa, ClassificationResult result, Taxonomy.Category category,
                                      String expectedCategory) {}
}
