package de.tum.claritypipeline.service;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.claritypipeline.model.classification.Classification;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.ClassificationProperties;
import de.tum.claritypipeline.model.core.Category;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.model.evaluation.Evaluation;
import de.tum.claritypipeline.model.relation.BelongsTo;
import de.tum.claritypipeline.model.relation.GeneratedBy;
import de.tum.claritypipeline.model.relation.HasClassification;
import de.tum.claritypipeline.model.relation.HasEvaluation;
import de.tum.clarityutils.ModelEvaluator;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.*;
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
     * Builder for constructing the graph ontology.
     */
    private final OntologyBuilder ontologyBuilder;
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
        this.client = new Neo4jClient(properties.getNeo4jCredentials());
        this.ontologyBuilder = new OntologyBuilder(client);
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
        ontologyBuilder.persistOntologyInGraph(properties);

        List<QA> qas = fetchQAs();
        Map<String, QA> unclassifiedQAs = filterUnclassifiedQAs(qas);

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
    private List<ClassificationTask> executeParallelClassification(Map<String, QA> unclassified) {
        List<ClassificationTask> tasks;

        log.info("Starting classification of {} unclassified QAs using {} threads.",
                 unclassified.size(), properties.getWorkerThreads());

        AtomicInteger counter = new AtomicInteger();
        try (ExecutorService executor = Executors.newFixedThreadPool(properties.getWorkerThreads())) {
            List<CompletableFuture<ClassificationTask>> futures =
                    unclassified.values().stream()
                                .map(qa -> CompletableFuture.supplyAsync(() -> {
                                    try {
                                        ClassificationRequest request = buildRequest(qa);
                                        ClassificationResult result = classifySingle(request);
                                        Category category = findAssignedCategory(result.getName());
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
            relations.add(createRelationObject(new BelongsTo(),
                                               task.result.getElementId(), task.category.getElementId()));
            relations.add(createRelationObject(new GeneratedBy(),
                                               task.result.getElementId(), properties.getElementId()));
        }

        client.batchCreateRelations(relations);
    }

    /**
     * Generate evaluation metrics for the classification run and store them in the database.
     */
    private void generateEvaluation() {
        log.info("Generating evaluation for classification run {}", properties.getVersion());
        String query = String.format("""
                                             MATCH (n:%s)--(cr:%s)--(c:%s)
                                             WHERE cr.version = '%s'
                                             AND c.name = '%s'
                                             RETURN n
                                             """,
                                     Neo4jNode.getLabel(ClassificationResult.class),
                                     Neo4jNode.getLabel(ClassificationProperties.class),
                                     Neo4jNode.getLabel(Classification.class),
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
                           return new String[]{result.getName(), qa.getClarityLabel()};
                       })
                       .filter(Objects::nonNull)
                       .toList();

        List<String> predictions = predictionsAndExpected.stream()
                                                         .map(arr -> arr[0])
                                                         .toList();

        List<String> expected = predictionsAndExpected.stream()
                                                      .map(arr -> arr[1])
                                                      .toList();

        List<String> labels = properties.getTaxonomy().getCategories()
                                        .stream()
                                        .map(Category::getName)
                                        .toList();

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

        Evaluation evaluation = Evaluation.builder()
                                          .accuracy(accuracy)
                                          .precision(precision)
                                          .recall(recall)
                                          .microF1(microF1)
                                          .macroF1(macroF1)
                                          .macroF1Rounded(Math.round(macroF1 * 100.0) / 100.0)
                                          .build();

        Evaluation existingEval = properties.getEvaluation(client);

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

    }

    // -------------------------------- Helper Methods --------------------------------

    /**
     * Filter out QAs that have already been classified in the context of the current ontology.
     *
     * @param qas The list of QAs to filter.
     * @return A map of unclassified QAs with their element IDs as keys.
     */
    private Map<String, QA> filterUnclassifiedQAs(List<QA> qas) {
        String qaIds = qas.stream()
                          .map(qa -> "'" + qa.getElementId() + "'")
                          .collect(Collectors.joining(","));

        String query = String.format("""
                                             MATCH (n:%s)-[:%s]->(:%s)-[:%s]->(cr:%s)
                                             WHERE elementId(n) IN [%s]
                                             AND elementId(cr) = '%s'
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
                  .collect(Collectors.toMap(QA::getElementId, qa -> qa));
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
                                    .question(qa.getQuestion())
                                    .context(qa.getInterviewQuestion() + "\n"
                                                     + qa.getInterviewAnswer())
                                    .taxonomy(properties.getTaxonomy())
                                    .answer(qa.getInterviewAnswer())
                                    .build();
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
    private Category findAssignedCategory(String name) {
        String normalizedName = name.replaceAll("[ _-]", "");
        for (Category category : properties.getTaxonomy().getCategories()) {
            String normalizedCategoryName = category.getName().replaceAll("[ _-]", "");
            if (normalizedCategoryName.equals(normalizedName)) {
                return category;
            }
        }
        for (Category category : properties.getTaxonomy().getCategories()) {
            String normalizedCategoryName = category.getName().replaceAll("[ _-]", "");
            if (normalizedCategoryName.contains(normalizedName)
                    || normalizedName.contains(normalizedCategoryName)) {
                return category;
            }
        }
        return null;
    }

    // -------------------------------- Inner Classes --------------------------------

    /**
     * A record representing a classification task, containing the QA, classification result, and category.
     */
    private record ClassificationTask(QA qa, ClassificationResult result, Category category) {}
}
