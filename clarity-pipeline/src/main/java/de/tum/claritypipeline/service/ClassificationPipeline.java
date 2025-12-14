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
     * Constructor to initialize the ClassificationPipeline with properties from a file.
     *
     * @param propertiesFilePath The path to the properties file.
     * @throws IOException If there is an error loading the properties file.
     */
    public ClassificationPipeline(String propertiesFilePath) throws IOException {
        this.properties = ClassificationProperties.load(propertiesFilePath);
        this.client = GlobalConfig.NEO4J_CLIENT;
    }

    // -------------------------------- Classification Logic --------------------------------

    /**
     * Classify a single question and answer pair with retries.
     *
     * @param classificationRequest The request containing question and answer.
     * @return The classification response from the classifier.
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
     * Execute the classification strategy defined in the properties.
     *
     * @param classificationRequest The request containing question and answer.
     * @return The classification result.
     */
    private ClassificationResult executeStrategy(
            ClassificationRequest classificationRequest
    ) {
        return properties.getStrategy().execute(classificationRequest);
    }

    // -------------------------------- Pipeline Entry --------------------------------

    /**
     * Classify QAs based on the query defined in the properties.
     * The classifications are stored in the Neo4j database.
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
     * Execute classification of QAs in parallel using multiple threads.
     *
     * @param unclassified A map of unclassified QAs with their element IDs as keys.
     * @return A list of classification tasks containing QAs, results, and categories.
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
                                        ClassificationRequest request = buildRequest(qa);
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
     * Batch save classification results and their relations to the database.
     *
     * @param tasks The list of classification tasks containing QAs, results, and categories.
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
     * Generate evaluation metrics for the classification run and store them in the database.
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
     * Filter out QAs that have already been classified in the context of the current ontology.
     *
     * @param qas The list of QAs to filter.
     * @return A map of unclassified QAs with their element IDs as keys.
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

    /**
     * Build a ClassificationRequest from a QA object.
     *
     * @param qa The QA object.
     * @return The constructed ClassificationRequest.
     */
    private ClassificationRequest buildRequest(QA qa) {
        return ClassificationRequest.builder()
                                    .qa(qa)
                                    .question(qa.getQuestion())
                                    .context(buildContext(qa.getInterviewQuestion(),
                                                          qa.getInterviewAnswer()))
                                    .taxonomy(properties.getTaxonomy())
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

    /**
     * Fetch QAs from the database based on the query defined in the properties.
     *
     * @return A list of QAs to be classified.
     */
    private List<QA> fetchQAs() {
        return client.executeQuery(properties.getQuery(), QA.class);
    }

    /**
     * Find the Category assigned to a given name in the ontology.
     *
     * @return The corresponding Category, or null if not found.
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
     * A record representing a classification task, containing the QA, classification result, and category.
     */
    private record ClassificationTask(QA qa, ClassificationResult result, Taxonomy.Category category) {}
}
