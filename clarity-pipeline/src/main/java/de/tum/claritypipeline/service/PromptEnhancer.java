package de.tum.claritypipeline.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import de.tum.clarityneo4j.core.Neo4jClient;
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
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

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

    /**
     * Constructs a PromptEnhancer with configuration loaded from a properties file.
     *
     * @param propertiesFilePath path to the prompt enhancing properties YAML file
     * @throws IOException if the properties file cannot be read or parsed
     */
    public PromptEnhancer(String propertiesFilePath) throws IOException {
        this.properties = PromptEnhancingProperties.load(propertiesFilePath);
        this.client = GlobalConfig.NEO4J_CLIENT;
    }

    /**
     * Writes the final enhanced prompt to a file if configured.
     * <p>
     * Supports two output formats:
     * <ul>
     *   <li><b>YAML</b> (.yaml, .yml): Wraps prompt in YAML structure with indentation</li>
     *   <li><b>Plain text</b>: Writes raw prompt content</li>
     * </ul>
     *
     * @param prompt the final enhanced prompt text
     */
    private void outputPromptToFile(String prompt, String path) {
        if (properties.getOutputPrompt() == null || properties.getOutputPrompt().isBlank()) {
            return;
        }
        try {
            if (path.endsWith(".yaml") || path.endsWith(".yml")) {
                StringBuilder yamlContent = new StringBuilder("prompt: |\n");
                for (String line : prompt.split("\n")) {
                    yamlContent.append("  ").append(line).append("\n");
                }
                SerializationUtils.writeStringToFile(path, yamlContent.toString());
            } else {
                SerializationUtils.writeStringToFile(path, prompt);
            }
        } catch (Exception e) {
            log.error("Failed to write prompt to file: {}", e.getMessage(), e);
        }
    }

    /**
     * Writes the final refined taxonomy to a YAML file if configured.
     * <p>
     * Serializes the complete Taxonomy object including all categories,
     * descriptions, examples, and mappings.
     *
     * @param taxonomy the final refined taxonomy structure
     */
    private void outputTaxonomyToFile(Taxonomy taxonomy, String path) {
        if (path == null || path.isBlank()) {
            return;
        }
        try {
            ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());
            yamlMapper.findAndRegisterModules();
            yamlMapper.writerWithDefaultPrettyPrinter();
            yamlMapper.writeValue(
                    new File(path),
                    taxonomy
            );
        } catch (IOException e) {
            log.error("Failed to write taxonomy to file: {}", e.getMessage(), e);
        }
    }

    /**
     * Classifies a single QA with retry logic and exponential backoff.
     * <p>
     * Implements the same retry mechanism as ClassificationPipeline:
     * <ul>
     *   <li>Attempts classification up to configured number of times</li>
     *   <li>Applies increasing delays between retries (attempt * 1000ms)</li>
     *   <li>Returns first successful result</li>
     * </ul>
     *
     * @param prompt                the classification prompt to use
     * @param classificationRequest the request containing question, answer, and taxonomy
     * @return classification result with predicted label and explanation
     * @throws RuntimeException if all retry attempts fail
     */
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

    /**
     * Builds the diagnosis prompt for failure mode analysis.
     * <p>
     * Constructs the prompt template with taxonomy information and
     * prepares placeholders for failure traces to be filled later.
     *
     * @param taxonomy the current taxonomy structure
     * @return diagnosis prompt template with placeholders
     */
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

    /**
     * Builds the patch prompt for improvement suggestions.
     * <p>
     * Constructs the prompt template with taxonomy information and
     * prepares placeholders for failure mode analysis to be filled later.
     *
     * @param taxonomy the current taxonomy structure
     * @return patch prompt template with placeholders
     */
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

    /**
     * Executes the complete prompt enhancement workflow.
     * <p>
     * Main entry point that orchestrates all enhancement phases:
     * <ol>
     *   <li>Fetches QAs from database using configured query</li>
     *   <li>Shuffles QAs for random sampling across iterations</li>
     *   <li>For each iteration:
     *     <ul>
     *       <li>Creates and saves iteration node</li>
     *       <li>Classifies N QAs with current prompt/taxonomy</li>
     *       <li>Persists results with iteration links</li>
     *       <li>Identifies and formats failures</li>
     *       <li>Performs LLM-based diagnosis</li>
     *       <li>Generates and applies patches</li>
     *     </ul>
     *   </li>
     *   <li>Exports final prompt and taxonomy to files</li>
     * </ol>
     * <p>
     * Stops early if no failures occur or no more QAs are available.
     */
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
                    iteration.setRevisedPrompt(currentClassificationPrompt);
                    client.updateNode(iteration);
                    if (!currentClassificationPrompt.contains("{question}") || !currentClassificationPrompt.contains(
                            "{context}")) {
                        log.error(
                                "The revised prompt does not contain required placeholders {question} or {context}. "
                                        + "Stopping enhancement process.");
                        break;
                    }
                } else {
                    log.warn("Patch result was null or empty, retaining current prompt.");
                }
            }

            log.info("Enhancement iteration {} completed with {} QAs.", i + 1, tasks.size());

            if (properties.isSaveTemporaryResults()) {
                log.debug("Saving temporary prompt and taxonomy after iteration {}.", i + 1);
                String classificationFileExtension = properties.getOutputPrompt() != null
                        ? properties.getOutputPrompt().substring(
                        properties.getOutputPrompt().lastIndexOf('.'))
                        : ".txt";
                String classificationFileBaseName = properties.getOutputPrompt() != null
                        ? properties.getOutputPrompt().substring(0,
                                                                 properties.getOutputPrompt().lastIndexOf('.'))
                        : "enhanced_prompt";
                outputPromptToFile(currentClassificationPrompt,
                                   classificationFileBaseName + "_iteration_" + (i + 1) + classificationFileExtension);
            }
        }
        outputPromptToFile(currentClassificationPrompt, properties.getOutputPrompt());

        log.info("Prompt enhancement process completed.");
    }

    /**
     * Filters classification tasks to only those with misclassifications.
     * <p>
     * Compares predicted category against expected ground truth label.
     * Returns tasks where prediction doesn't match expected (case-insensitive).
     *
     * @param tasks list of all classification tasks
     * @return list of tasks with incorrect predictions
     */
    private List<ClassificationTask> filterFailedClassifications(List<ClassificationTask> tasks) {
        return tasks.stream()
                    .filter(task -> task.expectedCategory != null
                            && !task.expectedCategory.equalsIgnoreCase(task.category.getName()))
                    .toList();
    }

    /**
     * Formats failure mode analysis into a readable string for the patch prompt.
     * <p>
     * Structures failure modes with:
     * <ul>
     *   <li>Failure mode name and description</li>
     *   <li>Prompt drivers with specific problematic lines</li>
     *   <li>Explanations of why each driver matters</li>
     * </ul>
     *
     * @param failureModesResult the LLM-generated failure analysis
     * @return formatted multi-line string for prompt inclusion
     */
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

    /**
     * Formats misclassification traces for the diagnosis prompt.
     * <p>
     * For each failed classification, includes:
     * <ul>
     *   <li>Original question and answer</li>
     *   <li>Assigned (incorrect) category</li>
     *   <li>Expected (correct) category</li>
     *   <li>Model's explanation for its choice</li>
     * </ul>
     *
     * @param tasks list of failed classification tasks
     * @return formatted multi-line string of failure traces
     */
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

    /**
     * Persists an iteration node and links it to the properties node.
     * <p>
     * Creates:
     * <ul>
     *   <li>PromptEnhancingIteration node in Neo4j</li>
     *   <li>HAS_ITERATION relationship from properties to iteration</li>
     * </ul>
     *
     * @param iteration the iteration object to save
     */
    private void saveIteration(PromptEnhancingIteration iteration) {
        client.saveNode(iteration);
        client.createRelation(createRelationObject(new HasIteration(),
                                                   properties.getElementId(), iteration.getElementId()));
    }

    /**
     * Batch saves classification results and creates all relevant relationships.
     * <p>
     * Creates nodes and relationships:
     * <ul>
     *   <li>ClassificationResult nodes (batch save)</li>
     *   <li>QA --[HAS_CLASSIFICATION]→ ClassificationResult</li>
     *   <li>ClassificationResult --[BELONGS_TO]→ Category</li>
     *   <li>ClassificationResult --[GENERATED_BY]→ Properties</li>
     *   <li>ClassificationResult --[BELONGS_TO_ITERATION]→ Iteration</li>
     * </ul>
     *
     * @param tasks     list of classification tasks with results
     * @param iteration the current enhancement iteration
     */
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
     * Creates a relationship object with specified start and end node IDs.
     * <p>
     * Helper method for batch relationship creation.
     *
     * @param relation    the relationship instance to configure
     * @param startNodeId element ID of the start node
     * @param endNodeId   element ID of the end node
     * @param <T>         relationship type extending Neo4jRelation
     * @return the configured relationship object
     */
    private <T extends Neo4jRelation> T createRelationObject(T relation, String startNodeId, String endNodeId) {
        relation.setStartNodeId(startNodeId);
        relation.setEndNodeId(endNodeId);
        return relation;
    }

    /**
     * Classifies a batch of QAs for a single enhancement iteration.
     * <p>
     * Workflow:
     * <ol>
     *   <li>Selects up to N QAs from unused set</li>
     *   <li>Classifies in parallel using thread pool</li>
     *   <li>Matches predictions to taxonomy categories</li>
     *   <li>Retrieves expected labels from ground truth</li>
     *   <li>Filters out failed classifications</li>
     * </ol>
     * <p>
     * Updates the unusedQAs list by removing processed QAs.
     *
     * @param unusedQAs mutable list of QAs not yet used (modified in place)
     * @param taxonomy  the current taxonomy structure
     * @param prompt    the current classification prompt
     * @param i         iteration number (for logging)
     * @return list of classification tasks with results and expected labels
     */
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

    /**
     * Builds a classification request for a single QA.
     * <p>
     * Constructs the request with:
     * <ul>
     *   <li>QA reference</li>
     *   <li>Extracted question text</li>
     *   <li>Formatted interview context</li>
     *   <li>Current taxonomy</li>
     *   <li>Answer text</li>
     * </ul>
     *
     * @param qa       the QA to classify
     * @param taxonomy the current taxonomy
     * @return complete classification request
     */
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

    /**
     * Builds interview context string from question and answer.
     * <p>
     * Formats as:
     * <pre>
     * Interviewer: [question]
     * Answer: [answer]
     * </pre>
     * Removes "Q. " prefix if present.
     *
     * @param interviewQuestion the interview question
     * @param interviewAnswer   the interview answer
     * @return formatted context string
     */
    private String buildContext(String interviewQuestion, String interviewAnswer) {
        StringBuilder contextBuilder = new StringBuilder();
        if (interviewQuestion.startsWith("Q. ")) {
            interviewQuestion = interviewQuestion.substring(3);
        }
        contextBuilder.append("Interviewer: ").append(interviewQuestion).append("\n");
        contextBuilder.append("Answer: ").append(interviewAnswer).append("\n");
        return contextBuilder.toString();
    }

    /**
     * Fetches QAs from Neo4j using the configured query.
     * <p>
     * The query is defined in PromptEnhancingProperties and typically
     * filters by dataset split or other criteria.
     *
     * @return list of QA records matching the query
     */
    private List<QA> fetchQAs() {
        return client.executeQuery(properties.getQuery(), QA.class);
    }

    /**
     * Finds the taxonomy category matching a predicted label name.
     * <p>
     * Implements flexible matching with normalization:
     * <ol>
     *   <li>Exact match after removing spaces, underscores, hyphens</li>
     *   <li>Substring match (contains) as fallback</li>
     * </ol>
     *
     * @param name the predicted category name
     * @return matching Category or null if no match found
     */
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

    /**
     * Retrieves the expected ground truth label for a QA.
     * <p>
     * Uses reflection to access the field specified by taxonomy's labelProperty.
     * Typically accesses "clarityLabel" or "evasionLabel" field.
     *
     * @param qa the QA record
     * @return ground truth label string or null if not available
     * @throws RuntimeException if reflection access fails
     */
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

    /**
     * Internal record bundling a complete classification task with ground truth.
     * <p>
     * Used to track classification results alongside their expected labels
     * for failure analysis.
     *
     * @param qa               the question-answer pair
     * @param result           the classification result from the model
     * @param category         the matched taxonomy category
     * @param expectedCategory the ground truth label from the dataset
     */
    private record ClassificationTask(QA qa, ClassificationResult result, Taxonomy.Category category,
                                      String expectedCategory) {}
}
