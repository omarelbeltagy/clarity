package de.tum.claritypipeline.service;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.claritypipeline.model.classification.Classification;
import de.tum.claritypipeline.model.config.ClassificationProperties;
import de.tum.claritypipeline.model.core.Category;
import de.tum.claritypipeline.model.core.Cluster;
import de.tum.claritypipeline.model.core.Taxonomy;
import de.tum.claritypipeline.model.relation.*;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Builds and ensures the ontology structure in the Neo4j database based on the provided classification properties.
 */
public class OntologyBuilder {
    /**
     * Logger instance for logging information and errors.
     */
    private final Logger log = org.slf4j.LoggerFactory.getLogger(OntologyBuilder.class);
    /**
     * Neo4j client for database interactions.
     */
    private final Neo4jClient client;

    /**
     * Initializes the OntologyBuilder with a Neo4j client.
     *
     * @throws IOException if there is an error initializing the Neo4j client.
     */
    public OntologyBuilder() throws IOException {
        this.client = new Neo4jClient();
    }

    /**
     * Initializes the OntologyBuilder with a Neo4j client using credentials from the specified file.
     *
     * @param neo4jCredentialsFile Path to the Neo4j credentials file.
     * @throws IOException if there is an error loading the credentials or initializing the Neo4j client.
     */
    public OntologyBuilder(String neo4jCredentialsFile) throws IOException {
        Neo4jCredentials neo4jCredentials = Neo4jCredentials.load(neo4jCredentialsFile);
        this.client = new Neo4jClient(neo4jCredentials);
    }

    /**
     * Initializes the OntologyBuilder with a Neo4j client using the provided credentials.
     *
     * @param neo4jCredentials Neo4j database credentials.
     */
    public OntologyBuilder(Neo4jCredentials neo4jCredentials) {
        this.client = new Neo4jClient(neo4jCredentials);
    }

    /**
     * Initializes the OntologyBuilder with a Neo4j client using an existing Neo4jClient
     *
     * @param neo4jClient Neo4jClient
     */
    public OntologyBuilder(Neo4jClient neo4jClient) {
        this.client = neo4jClient;
    }

    /**
     * Builds the ontology in the Neo4j database based on the provided classification properties.
     *
     * @param properties The classification properties defining the ontology structure.
     */
    public void persistOntologyInGraph(ClassificationProperties properties) {

        findOrCreateCluster(properties);
        findOrCreateClassification(properties);
        findOrCreateRun(properties);
        ensureTaxonomy(properties);

        if (properties.getCluster() != null) {
            if (client.findRelation(properties.getClassification().getElementId(),
                                    properties.getCluster().getElementId(),
                                    BelongsTo.class) == null) {
                BelongsTo belongsToRelation = new BelongsTo();
                belongsToRelation.setStartNodeId(properties.getClassification().getElementId());
                belongsToRelation.setEndNodeId(properties.getCluster().getElementId());
                client.createRelation(belongsToRelation);
            }
        }

        if (client.findRelation(properties.getClassification().getElementId(),
                                properties.getElementId(), HasRun.class) == null) {
            HasRun hasRunRelation = new HasRun();
            hasRunRelation.setStartNodeId(properties.getClassification().getElementId());
            hasRunRelation.setEndNodeId(properties.getElementId());
            client.createRelation(hasRunRelation);
        }

        if (client.findRelation(properties.getElementId(),
                                properties.getTaxonomy().getElementId(), HasTaxonomy.class) == null) {
            HasTaxonomy hasTaxonomyRelation = new HasTaxonomy();
            hasTaxonomyRelation.setStartNodeId(properties.getElementId());
            hasTaxonomyRelation.setEndNodeId(properties.getTaxonomy().getElementId());
            client.createRelation(hasTaxonomyRelation);
        }
    }

    /**
     * Finds an existing Classification root node by name or creates a new one if it doesn't exist.
     *
     * @param properties The classification properties containing the name.
     */
    private synchronized void findOrCreateClassification(ClassificationProperties properties) {
        Classification classification = client.findOrCreateNode(
                Map.of("name", properties.getClassification().getName()),
                Classification.class,
                () -> Classification.builder().name(properties.getClassification().getName()).build()
        );
        if (classification.getElementId() == null) {
            throw new RuntimeException("Failed to create or find Classification node for "
                                               + properties.getClassification().getName());
        }
        properties.getClassification().setElementId(classification.getElementId());
    }

    /**
     * Finds an existing Cluster node by name or creates a new one if it doesn't exist.
     *
     * @param properties The classification properties containing the cluster information.
     */
    private synchronized void findOrCreateCluster(ClassificationProperties properties) {
        if (properties.getCluster() == null) {
            return;
        }
        Cluster cluster = client.findOrCreateNode(
                Map.of("name", properties.getCluster().getName()),
                Cluster.class,
                () -> Cluster.builder().name(properties.getCluster().getName()).build()
        );
        if (cluster.getElementId() == null) {
            throw new RuntimeException("Failed to create or find Cluster node for "
                                               + properties.getCluster().getName());
        }
        properties.getCluster().setElementId(cluster.getElementId());
    }

    /**
     * Finds an existing ClassificationProperties node by the version. If the version does not exist, creates a new
     * ClassificationProperties node.
     *
     * @param properties The classification properties containing the version, prompt, and model.
     */
    private synchronized void findOrCreateRun(ClassificationProperties properties) {
        try {
            String query = String.format("""
                                                 MATCH (c:%s)-[:%s]->(n:%s)
                                                 WHERE c.name = '%s'
                                                 AND n.version = '%s'
                                                 RETURN n
                                                 """,
                                         Neo4jNode.getLabel(Classification.class),
                                         Neo4jRelation.getType(HasRun.class),
                                         Neo4jNode.getLabel(ClassificationProperties.class),
                                         properties.getName(),
                                         properties.getVersion());

            ClassificationProperties existing = client.executeQuery(query, ClassificationProperties.class)
                                                      .stream().findFirst().orElse(null);
            if (existing != null && existing.getElementId() != null) {
                log.info("Found existing {} node for {} with version {}",
                         Neo4jNode.getLabel(ClassificationProperties.class), properties.getName(),
                         properties.getVersion());
                properties.setElementId(existing.getElementId());
                return;
            }
            client.saveNode(properties);
            if (properties.getElementId() == null) {
                throw new RuntimeException(
                        "Failed to create " + Neo4jNode.getLabel(ClassificationProperties.class) + " node for "
                                + properties.getName());
            }
        } catch (Exception e) {
            log.error("Error while creating classification run node for {}", properties.getName(), e);
            throw new RuntimeException(e);
        }
    }

    /**
     * Ensures that all Categories defined in the Category Ontology exists in the graph
     *
     * @param properties The classification properties
     */
    private synchronized void ensureTaxonomy(ClassificationProperties properties) {
        ensureTaxonomyRootNode(properties);
        String query = String.format("""
                                             MATCH (cs:%s)-[r:%s]->(n:%s)
                                             WHERE cs.name = '%s'
                                             RETURN n
                                             """,
                                     Neo4jNode.getLabel(Taxonomy.class),
                                     Neo4jRelation.getType(HasCategory.class),
                                     Neo4jNode.getLabel(Category.class),
                                     properties.getTaxonomy().getName()
        );
        List<Category> existingInGraph = client.executeQuery(query, Category.class);
        List<Category> ontologyCategories = new ArrayList<>(List.copyOf(properties.getTaxonomy().getCategories()));

        if (properties.getTaxonomy().getMapping() != null) {
            String mappingQuery = """
                    MATCH(taxonomy:%s)-[:%s]->(n:%s)
                    WHERE elementId(taxonomy) = $taxonomyId
                    RETURN n
                    """.formatted(
                    Neo4jNode.getLabel(Taxonomy.class),
                    Neo4jRelation.getType(HasMapping.class),
                    Neo4jNode.getLabel(Taxonomy.Mapping.class)
            );
            Map<String, Object> params = Map.of("taxonomyId", properties.getTaxonomy().getElementId());
            Taxonomy.Mapping mapping = client.executeQuery(mappingQuery, params,
                                                           Taxonomy.Mapping.class).stream()
                                             .findFirst().orElse(null);
            if (mapping != null) {
                log.info("Mapping node for Taxonomy already exists. Updating it.");
                properties.getTaxonomy().getMapping().setElementId(mapping.getElementId());
                client.updateNode(properties.getTaxonomy().getMapping());
            } else {
                log.info("Mapping node for Taxonomy is missing. Creating it.");
                client.saveNode(properties.getTaxonomy().getMapping());
                HasMapping hasMapping = new HasMapping();
                hasMapping.setStartNodeId(properties.getTaxonomy().getElementId());
                hasMapping.setEndNodeId(properties.getTaxonomy().getMapping().getElementId());
                client.createRelation(hasMapping);
            }
        }

        for (Category node : existingInGraph) {
            Category inOntologyFile = ontologyCategories.stream().filter(category -> category.getName()
                                                                                             .equals(node.getName()))
                                                        .findFirst().orElse(null);
            if (inOntologyFile != null) {
                if (!inOntologyFile.getDescription().equals(node.getDescription())) {
                    log.info("Description was updated for category: {} from {} to  {}", node.getName(),
                             node.getDescription(), inOntologyFile.getDescription());
                    node.setDescription(inOntologyFile.getDescription());
                    client.updateNode(node);
                }
                properties.getTaxonomy().getCategories().stream()
                          .filter(category -> category.getName().equals(node.getName()))
                          .findFirst().ifPresent(category ->
                                                         category.setElementId(node.getElementId()));
                log.info("Cached existing category node for {} with elementId {}", node.getName(),
                         node.getElementId());
                ontologyCategories.remove(inOntologyFile);
            }
        }

        for (Category missing : ontologyCategories) {
            log.info("Category node {} is missing. Creating it.", missing.getName());
            client.saveNode(missing);
            properties.getTaxonomy().getCategories().stream()
                      .filter(category -> category.getName().equals(missing.getName()))
                      .findFirst().ifPresent(
                              category -> category.setElementId(missing.getElementId())
                      );
            HasCategory hasCategory = new HasCategory();
            hasCategory.setStartNodeId(properties.getTaxonomy().getElementId());
            hasCategory.setEndNodeId(missing.getElementId());
            client.createRelation(hasCategory);
        }

        for (Category category : properties.getTaxonomy().getCategories()) {
            String exampleQuery = """
                    MATCH(category:%s)-[:%s]->(n:%s)
                    WHERE elementId(category) = $categoryId
                    RETURN n
                    """.formatted(
                    Neo4jNode.getLabel(Category.class),
                    Neo4jRelation.getType(HasExample.class),
                    Neo4jNode.getLabel(Category.TaxonomyExample.class)
            );
            Map<String, Object> params = Map.of("categoryId", category.getElementId());
            List<Category.TaxonomyExample> examples = client.executeQuery(exampleQuery, params,
                                                                          Category.TaxonomyExample.class);
            if (!examples.isEmpty()) {
                log.info("Removing {} existing examples nodes for {}", examples.size(), category.getName());
                examples.forEach(client::deleteNode);
            }
            if (category.getExamples() != null && !category.getExamples().isEmpty()) {
                log.info("Creating {} example nodes for {}.", category.getExamples().size(), category.getName());
                for (Category.TaxonomyExample example : category.getExamples()) {
                    client.saveNode(example);
                    HasExample hasExample = new HasExample();
                    hasExample.setStartNodeId(category.getElementId());
                    hasExample.setEndNodeId(example.getElementId());
                    client.createRelation(hasExample);
                }
            }
        }
    }

    /**
     * Ensures that the Taxonomy root node exists in the graph
     *
     * @param properties The classification properties
     */
    private synchronized void ensureTaxonomyRootNode(ClassificationProperties properties) {
        try {
            String query = String.format("""
                                                 MATCH (n:%s)
                                                 WHERE n.name = '%s'
                                                 RETURN n
                                                 """,
                                         Neo4jNode.getLabel(Taxonomy.class),
                                         properties.getTaxonomy().getName()
            );
            Taxonomy categorySet = client.executeQuery(query, Taxonomy.class).stream().findFirst().orElse(null);
            if (categorySet != null) {
                properties.getTaxonomy().setElementId(categorySet.getElementId());
                if (!Objects.equals(categorySet.getDescription(), properties.getTaxonomy().getDescription())) {
                    log.info("{} description for {} was updated", Neo4jNode.getLabel(Taxonomy.class),
                             properties.getTaxonomy().getName());
                    client.updateNode(properties.getTaxonomy());
                }
                return;
            }
            log.info("Creating new {} node for {}", Neo4jNode.getLabel(Taxonomy.class), properties.getName());
            client.saveNode(properties.getTaxonomy());
        } catch (Exception e) {
            log.error("Error while creating classification category nodes for {}", properties.getName(), e);
            throw new RuntimeException(e);
        }
    }
}
