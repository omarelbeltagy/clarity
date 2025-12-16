package de.tum.claritypipeline.service;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.ClassificationProperties;
import de.tum.claritypipeline.model.config.GlobalConfig;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.model.core.Taxonomy;
import de.tum.claritypipeline.model.relation.BelongsTo;
import de.tum.claritypipeline.model.relation.GeneratedBy;
import de.tum.claritypipeline.model.relation.HasClassification;
import de.tum.claritypipeline.model.relation.HasEvaluation;
import de.tum.claritypipeline.strategy.BestGuessStrategy;
import de.tum.claritypipeline.strategy.ClassificationStrategy;
import de.tum.claritypipeline.utils.PipelineUtils;
import de.tum.clarityutils.ModelEvaluator;
import org.slf4j.Logger;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Classification pipeline for processing question-answer pairs using configurable classification strategies.
 * <p>
 * This pipeline orchestrates the entire classification workflow including:
 * <ul>
 *   <li>Fetching QA pairs from Neo4j database based on configured queries</li>
 *   <li>Filtering already classified QAs to avoid duplicate processing</li>
 *   <li>Executing parallel classification using configurable thread pools</li>
 *   <li>Persisting classification results and relationships to Neo4j</li>
 *   <li>Generating evaluation metrics (accuracy, precision, recall, F1 scores)</li>
 * </ul>
 *
 * <h2>Classification Process</h2>
 * The pipeline follows these main steps:
 * <ol>
 *   <li><b>Fetch QAs</b>: Retrieve QA pairs from database using configured query</li>
 *   <li><b>Filter Unclassified</b>: Exclude QAs already classified by this configuration</li>
 *   <li><b>Parallel Classification</b>: Execute classification strategy across multiple threads</li>
 *   <li><b>Category Matching</b>: Map predicted labels to taxonomy categories</li>
 *   <li><b>Batch Persistence</b>: Save results and relationships in batches for efficiency</li>
 *   <li><b>Evaluation</b>: Generate and persist performance metrics</li>
 * </ol>
 *
 * <h2>Retry Mechanism</h2>
 * Classifications support automatic retries with exponential backoff to handle transient failures.
 * The number of retry attempts is configured via {@link ClassificationProperties#getAttempts()}.
 *
 * <h2>Thread Safety</h2>
 * The pipeline uses a fixed thread pool executor for parallel processing. The number of worker
 * threads is configurable via {@link ClassificationProperties#getWorkerThreads()}.
 *
 * @see ClassificationProperties
 * @see ClassificationStrategy
 * @see ClassificationResult
 */
public class ClassificationPipeline {
    /**
     * Logger for logging classification pipeline activities.
     */
    private final Logger log = org.slf4j.LoggerFactory.getLogger(ClassificationPipeline.class);
    /**
     * Properties for configuring the classification pipeline.
     */
    private final ClassificationProperties properties;
    /**
     * Neo4j client for database interactions.
     */
    private final Neo4jClient client;

    /**
     * Constructs a ClassificationPipeline with configuration loaded from a properties file.
     *
     * @param propertiesFilePath the path to the classification properties file
     * @throws IOException if the properties file cannot be read or parsed
     */
    public ClassificationPipeline(String propertiesFilePath) throws IOException {
        this.properties = ClassificationProperties.load(propertiesFilePath);
        this.client = GlobalConfig.NEO4J_CLIENT;
    }

    // -------------------------------- Classification Logic --------------------------------

    /**
     * Classifies a single question-answer pair using the configured strategy with automatic retry support.
     * <p>
     * This method implements an exponential backoff retry mechanism. If classification fails,
     * it will retry up to the configured number of attempts with increasing delays between tries.
     * The delay increases linearly with each retry attempt (attempt_number * 1000ms).
     *
     * @param classificationRequest the request containing the question, answer, and taxonomy
     * @return the classification result containing predicted label and metadata
     * @throws RuntimeException if all retry attempts fail
     * @see ClassificationProperties#getAttempts()
     */
    public ClassificationResult classifySingle(
            ClassificationRequest classificationRequest
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
                ClassificationResult classificationResult = executeStrategy(classificationRequest);
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
     * Executes the configured classification strategy for a single request.
     * <p>
     * Delegates to the strategy specified in {@link ClassificationProperties#getStrategy()}.
     * Different strategies implement different classification approaches (single model, multi-model,
     * judgement-based, etc.).
     *
     * @param classificationRequest the request containing question, answer, and taxonomy
     * @return the classification result from the strategy
     * @see ClassificationStrategy
     */
    private ClassificationResult executeStrategy(
            ClassificationRequest classificationRequest
    ) {
        return properties.getStrategy().execute(classificationRequest);
    }

    // -------------------------------- Pipeline Entry --------------------------------

    /**
     * Executes the complete classification pipeline.
     * <p>
     * This is the main entry point that orchestrates the entire classification workflow:
     * <ol>
     *   <li>Fetches all QA pairs matching the configured query</li>
     *   <li>Filters out QAs already classified by this configuration</li>
     *   <li>Executes parallel classification using worker threads</li>
     *   <li>Batch saves results and relationships to Neo4j</li>
     *   <li>Generates and persists evaluation metrics</li>
     * </ol>
     * <p>
     * The method logs progress information and handles the complete lifecycle from data retrieval
     * to evaluation metric generation.
     *
     * @see #fetchQAs()
     * @see #filterUnclassifiedQAs(List)
     * @see #executeParallelClassification(List)
     * @see #batchSaveClassifications(List)
     * @see #generateEvaluation()
     */
    public void classify() {
        List<QA> allQAs = fetchQAs();
        List<QA> unclassifiedQAs = filterUnclassifiedQAs(allQAs);

        List<ClassificationTask> tasks = executeParallelClassification(unclassifiedQAs);
        log.info("Classification completed. Classified {} / {} QAs.",
                 tasks.size(), unclassifiedQAs.size());

        if (!tasks.isEmpty()) {
            batchSaveClassifications(tasks);
        }

        generateEvaluation();
    }

    // -------------------------------- Pipeline Steps --------------------------------

    /**
     * Executes classification of QAs in parallel using a fixed thread pool.
     * <p>
     * This method distributes classification tasks across multiple threads for improved performance.
     * For each QA:
     * <ol>
     *   <li>Builds a classification request with taxonomy information</li>
     *   <li>Executes the classification strategy</li>
     *   <li>Matches the predicted label to a taxonomy category</li>
     *   <li>Logs progress information</li>
     * </ol>
     * <p>
     * Failed classifications (those returning null or without valid categories) are filtered out.
     * The method blocks until all classification tasks complete.
     *
     * @param unclassified list of QA pairs to classify
     * @return list of classification tasks containing QAs, results, and matched categories
     * @see ClassificationProperties#getWorkerThreads()
     */
    private List<ClassificationTask> executeParallelClassification(List<QA> unclassified) {
        List<ClassificationTask> tasks;

        log.info("Starting classification for {} ({}) of {} unclassified QAs using {} threads.", properties.getName(),
                 properties.getVersion(),
                 unclassified.size(), properties.getWorkerThreads());

        AtomicInteger counter = new AtomicInteger();
        try (ExecutorService executor = Executors.newFixedThreadPool(properties.getWorkerThreads())) {
            List<CompletableFuture<ClassificationTask>> futures =
                    unclassified.stream()
                                .map(qa -> CompletableFuture.supplyAsync(() -> {
                                    try {
                                        ClassificationRequest request = PipelineUtils.buildRequest(qa,
                                                                                                   properties.getTaxonomy());
                                        ClassificationResult result = classifySingle(request);
                                        Taxonomy.Category category = findAssignedCategory(result.getName());
                                        if (category != null) {
                                            result.setName(category.getName());
                                        }
                                        log.info("Classified QA as {} ({}/{})",
                                                 category != null ? category.getName() : "UNKNOWN",
                                                 counter.incrementAndGet(),
                                                 unclassified.size());
                                        return new ClassificationTask(qa, result, category);
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
            tasks.forEach(task -> task.result.setName(task.category.getName()));
        }
        return tasks;
    }

    /**
     * Persists classification results and their relationships to Neo4j in batches.
     * <p>
     * Creates the following relationships:
     * <ul>
     *   <li>{@link HasClassification}: QA → ClassificationResult</li>
     *   <li>{@link BelongsTo}: ClassificationResult → Category</li>
     *   <li>{@link GeneratedBy}: ClassificationResult → ClassificationProperties</li>
     * </ul>
     * <p>
     * Batch operations are used for efficiency when handling large numbers of classifications.
     *
     * @param tasks list of classification tasks containing results and categories
     */
    private void batchSaveClassifications(List<ClassificationTask> tasks) {
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
        }

        client.batchCreateRelations(relations);
    }

    /**
     * Generates and persists evaluation metrics for the classification run.
     * <p>
     * This method:
     * <ol>
     *   <li>Retrieves all classification results for this configuration version</li>
     *   <li>Fetches corresponding QA pairs with ground truth labels</li>
     *   <li>Handles label mapping if configured in the taxonomy</li>
     *   <li>Calculates metrics using {@link ModelEvaluator}:
     *     <ul>
     *       <li>Accuracy: overall correctness</li>
     *       <li>Precision: positive predictive value per class</li>
     *       <li>Recall: sensitivity per class</li>
     *       <li>Micro F1: harmonic mean weighted by support</li>
     *       <li>Macro F1: unweighted average F1 across classes</li>
     *     </ul>
     *   </li>
     *   <li>Persists or updates evaluation results in Neo4j</li>
     * </ol>
     * <p>
     * If evaluation already exists for this classification run, values are updated.
     * Errors during evaluation are logged but don't fail the pipeline.
     */
    private void generateEvaluation() {
        log.info("Generating evaluation for classification run {} of {}", properties.getVersion(),
                 properties.getName());
        String query = String.format("""
                                             MATCH (n:%s)--(cr:%s)--(c:%s)
                                             WHERE cr.version = '%s'
                                             AND c.name = '%s'
                                             RETURN n
                                             """,
                                     Neo4jNode.getLabel(ClassificationResult.class),
                                     Neo4jNode.getLabel(ClassificationProperties.class),
                                     Neo4jNode.getLabel(ClassificationProperties.Classification.class),
                                     properties.getVersion(),
                                     properties.getClassification().getName());
        List<ClassificationResult> results = client.executeQuery(query,
                                                                 ClassificationResult.class);
        log.info("Found {} classification results for evaluation", results.size());
        List<String[]> predictionsAndExpected =
                results.parallelStream()
                       .map(result -> {
                           String findQAQuery = String.format("""
                                                                      MATCH (cr:%s)--(n:%s)
                                                                      WHERE elementId(cr) = '%s'
                                                                      RETURN n
                                                                      """,
                                                              Neo4jNode.getLabel(ClassificationResult.class),
                                                              Neo4jNode.getLabel(QA.class),
                                                              result.getElementId());

                           QA qa = client.executeQuery(findQAQuery, QA.class)
                                         .stream()
                                         .findFirst()
                                         .orElse(null);

                           if (qa == null) {
                               log.warn("No QA found for classification result {}. Could not generate evaluation",
                                        result.getElementId());
                               return null;
                           }
                           String predictedLabel;
                           if (!(properties.getStrategy() instanceof BestGuessStrategy)
                                   && properties.getTaxonomy().getMapping() != null && properties.getTaxonomy()
                                                                                                 .getMapping()
                                                                                                 .isEnabled()) {
                               Taxonomy.Category category = properties.getTaxonomy().getCategories().stream()
                                                                      .filter(c ->
                                                                                      c.getName()
                                                                                       .equals(result.getName()))
                                                                      .findFirst()
                                                                      .orElse(null);
                               if (category != null) {
                                   predictedLabel = category.getMapTo();
                               } else {
                                   return null;
                               }
                           } else {
                               predictedLabel = result.getName();
                           }
                           String propertyLabel =
                                   ((properties.getStrategy() instanceof BestGuessStrategy)
                                           || (properties.getTaxonomy().getMapping() != null && properties.getTaxonomy()
                                                                                                          .getMapping()
                                                                                                          .isEnabled()))
                                           ? properties.getTaxonomy().getMapping().getLabelProperty()
                                           : properties.getTaxonomy().getLabelProperty();
                           String expectedLabel;
                           try {
                               Field field = qa.getClass().getDeclaredField(propertyLabel);
                               field.setAccessible(true);
                               Object value = field.get(qa);
                               if (value == null) {
                                   return null;
                               }
                               expectedLabel = value.toString();
                           } catch (NoSuchFieldException | IllegalAccessException e) {
                               throw new RuntimeException(e);
                           }
                           if (predictedLabel != null && expectedLabel != null) {
                               return new String[]{predictedLabel, expectedLabel};
                           }
                           return null;
                       })
                       .filter(Objects::nonNull)
                       .toList();

        List<String> predictions = predictionsAndExpected.stream()
                                                         .map(arr -> arr[0])
                                                         .toList();

        List<String> expected = predictionsAndExpected.stream()
                                                      .map(arr -> arr[1])
                                                      .toList();

        List<String> labels;
        if (properties.getTaxonomy().getMapping() != null && properties.getTaxonomy().getMapping().isEnabled()) {
            labels = properties.getTaxonomy().getMapping().getLabels();
        } else {
            labels = properties.getTaxonomy().getCategories()
                               .stream()
                               .map(Taxonomy.Category::getName)
                               .toList();
        }

        try {
            ModelEvaluator evaluator = new ModelEvaluator(labels, predictions, expected);
            log.info("Evaluation Results:");
            double accuracy = evaluator.accuracy();
            log.info("Accuracy: {}", String.format("%.2f", accuracy * 100));
            double precision = evaluator.precision();
            log.info("Precision: {}", String.format("%.2f", precision * 100));
            double recall = evaluator.recall();
            log.info("Recall: {}", String.format("%.2f", recall * 100));
            double microF1 = evaluator.microF1();
            log.info("Micro F1 Score: {}", String.format("%.2f", microF1 * 100));
            double macroF1 = evaluator.macroF1();
            log.info("Macro F1 Score: {}", String.format("%.2f", macroF1 * 100));

            ClassificationProperties.Evaluation evaluation = ClassificationProperties.Evaluation.builder()
                                                                                                .accuracy(accuracy)
                                                                                                .precision(precision)
                                                                                                .recall(recall)
                                                                                                .microF1(microF1)
                                                                                                .macroF1(macroF1)
                                                                                                .macroF1Rounded(
                                                                                                        Math.round(
                                                                                                                macroF1
                                                                                                                        * 100.0)
                                                                                                                / 100.0)
                                                                                                .build();

            ClassificationProperties.Evaluation existingEval = properties.getEvaluation(client);

            if (existingEval == null) {
                client.saveNode(evaluation);
                HasEvaluation hasEvaluation = createRelationObject(new HasEvaluation(), properties.getElementId(),
                                                                   evaluation.getElementId());
                client.createRelation(hasEvaluation);
            } else {
                log.info("Evaluation already exists for classification run {}. Updating values.",
                         properties.getVersion());
                evaluation.setElementId(existingEval.getElementId());
                client.updateNode(evaluation);
            }
        } catch (Exception e) {
            log.error("Error while evaluating classification run {}", properties.getVersion(), e);
        }

    }

    // -------------------------------- Helper Methods --------------------------------

    /**
     * Filters out QA pairs that have already been classified by this configuration.
     * <p>
     * Queries Neo4j to find QAs with existing HasClassification relationships
     * pointing to ClassificationResults generated by this ClassificationProperties instance.
     * Only unclassified QAs are returned to avoid duplicate processing.
     *
     * @param qas the complete list of QA pairs
     * @return list of QAs not yet classified by this configuration
     */
    private List<QA> filterUnclassifiedQAs(List<QA> qas) {
        String qaIds = qas.stream()
                          .map(qa -> "'" + qa.getElementId() + "'")
                          .collect(Collectors.joining(","));

        String query = String.format("""
                                             MATCH (n:%s)-[:%s]->(:%s)-[:%s]->(cp:%s)
                                             WHERE elementId(n) IN [%s]
                                             AND elementId(cp) = '%s'
                                             RETURN n
                                             """,
                                     Neo4jNode.getLabel(QA.class),
                                     Neo4jRelation.getType(HasClassification.class),
                                     Neo4jNode.getLabel(ClassificationResult.class),
                                     Neo4jRelation.getType(GeneratedBy.class),
                                     Neo4jNode.getLabel(ClassificationProperties.class),
                                     qaIds,
                                     properties.getElementId()
        );

        Set<String> classified = client.executeQuery(query, QA.class)
                                       .stream().map(QA::getElementId).collect(Collectors.toSet());

        return qas.stream()
                  .filter(qa -> !classified.contains(qa.getElementId()))
                  .collect(Collectors.toCollection(ArrayList::new));
    }

    /**
     * Creates a relationship object with specified start and end node IDs.
     * <p>
     * Helper method to populate relationship objects for batch creation in Neo4j.
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
     * Fetches QA pairs from Neo4j database based on the configured query.
     * <p>
     * The query is defined in {@link ClassificationProperties#getQuery()} and typically
     * filters QAs by dataset, domain, or other criteria.
     *
     * @return list of QA pairs matching the query criteria
     */
    private List<QA> fetchQAs() {
        return client.executeQuery(properties.getQuery(), QA.class);
    }

    /**
     * Finds the taxonomy category matching a predicted label name.
     * <p>
     * Implements flexible matching with normalization (removing spaces, underscores, hyphens):
     * <ol>
     *   <li>Exact match after normalization</li>
     *   <li>Substring match (contains) as fallback</li>
     * </ol>
     * <p>
     * For {@link BestGuessStrategy}, matches against mapping labels.
     * For other strategies, matches against taxonomy category names.
     *
     * @param name the predicted label name from the classification model
     * @return the matching Category, or null if no match found
     * @see Taxonomy.Category
     */
    private Taxonomy.Category findAssignedCategory(String name) {
        String normalizedName = name.replaceAll("[ _-]", "");
        if (properties.getStrategy() instanceof BestGuessStrategy) {
            for (String category : properties.getTaxonomy().getMapping().getLabels()) {
                String normalizedCategoryName = category.replaceAll("[ _-]", "");
                if (normalizedCategoryName.equals(normalizedName)) {
                    return Taxonomy.Category.builder().name(category).build();
                }
            }
            for (String category : properties.getTaxonomy().getMapping().getLabels()) {
                String normalizedCategoryName = category.replaceAll("[ _-]", "");
                if (normalizedCategoryName.contains(normalizedName)
                        || normalizedName.contains(normalizedCategoryName)) {
                    return Taxonomy.Category.builder().name(category).build();
                }
            }
        } else {
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
        }
        return null;
    }

    // -------------------------------- Inner Classes --------------------------------

    /**
     * Internal record representing a complete classification task.
     * <p>
     * Bundles together the QA pair, classification result, and matched taxonomy category
     * for efficient processing and persistence.
     *
     * @param qa       the question-answer pair that was classified
     * @param result   the classification result containing predicted label and metadata
     * @param category the matched taxonomy category
     */
    private record ClassificationTask(QA qa, ClassificationResult result, Taxonomy.Category category) {}
}
