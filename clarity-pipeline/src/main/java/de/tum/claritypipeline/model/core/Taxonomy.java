package de.tum.claritypipeline.model.core;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.claritypipeline.model.config.GlobalConfig;
import de.tum.claritypipeline.model.relation.HasCategory;
import de.tum.claritypipeline.model.relation.HasExample;
import de.tum.claritypipeline.model.relation.HasMapping;
import de.tum.clarityutils.AfterDeserialization;
import lombok.*;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Represents a taxonomy consisting of named categories used for classification.
 *
 * <p>The taxonomy can be loaded from a YAML file via {@link #load(String)}.
 */
@Node(label = "Taxonomy")
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@Builder
public class Taxonomy extends Neo4jNode implements Serializable {

    /**
     * The list of categories defined in this taxonomy.
     *
     * <p>Not persisted as separate properties on the Neo4j node; used to validate/lookup labels.
     */
    @Neo4jIgnore
    @JsonProperty("categories")
    @JsonPropertyDescription("List of categories in the taxonomy")
    List<Category> categories;
    /**
     * The name of the taxonomy.
     */
    @JsonProperty("name")
    @JsonPropertyDescription("Name of the taxonomy")
    private String name;
    /**
     * A human-readable description of the taxonomy and its purpose.
     */
    @JsonProperty("description")
    @JsonPropertyDescription("Description of the taxonomy")
    private String description;

    @JsonProperty("mapping")
    @JsonPropertyDescription("Mapping properties to another taxonomy")
    @Neo4jIgnore
    private Mapping mapping;

    @JsonProperty("version")
    @JsonPropertyDescription("The taxonomy version")
    private String version;

    @JsonProperty("label-property")
    @JsonPropertyDescription("The property key of the QA Node where the ground truth is found")
    private String labelProperty;

    @AfterDeserialization
    public void initialize() {
        Taxonomy existing = GlobalConfig.NEO4J_CLIENT.findNode(toPropertiesMap(), Taxonomy.class);

        if (existing == null || !allRelationsExist(existing)) {
            saveTaxonomyWithCategories();
            createRelationIfNeeded(mapping, new HasMapping());
            return;
        }

        validateCategoriesMatch(existing);
        createRelationIfNeeded(mapping, new HasMapping());
        this.setElementId(existing.getElementId());
    }

    private boolean allRelationsExist(Taxonomy existingNode) {
        return mapping == null
                || GlobalConfig.NEO4J_CLIENT.findRelation(existingNode.getElementId(), mapping.getElementId(),
                                                          HasMapping.class)
                != null;
    }

    /**
     * Load a Taxonomy instance from a YAML file.
     *
     * @param path the filesystem path to the taxonomy YAML file
     * @return the deserialized Taxonomy
     * @throws IOException if the file cannot be read or required fields (e.g., name) are missing
     */
    public static Taxonomy load(String path) throws IOException {
        if (path == null || path.isEmpty()) {
            throw new IOException("No path specified for Taxonomy file.");
        }
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        Taxonomy taxonomy = mapper.readValue(new File(path),
                                             Taxonomy.class);
        if (taxonomy.getName() == null || taxonomy.getName().isEmpty()) {
            throw new IOException("Missing name for taxonomy.");
        }
        return taxonomy;
    }

    private <T extends Neo4jRelation, N extends Neo4jNode> void createRelationIfNeeded(
            N targetNode, T relation) {
        if (targetNode == null) return;
        relation.setStartNodeId(this.getElementId());
        relation.setEndNodeId(targetNode.getElementId());
        GlobalConfig.NEO4J_CLIENT.createRelation(relation);
    }

    @Getter
    @Setter
    @Node(label = "Mapping")
    public static class Mapping extends Neo4jNode {
        @JsonProperty("enabled")
        @Neo4jIgnore
        private boolean enabled;
        @JsonProperty("labels")
        private List<String> labels = new ArrayList<>();
        @JsonProperty("label-property")
        @JsonPropertyDescription("The property key of the QA Node where the ground truth is found")
        private String labelProperty;

        @AfterDeserialization
        public void initialize() {
            Mapping existing = GlobalConfig.NEO4J_CLIENT.findNode(toPropertiesMap(), Mapping.class);
            if (existing != null) {
                this.setElementId(existing.getElementId());
                return;
            }
            GlobalConfig.NEO4J_CLIENT.saveNode(this);
        }
    }

    /**
     * A single taxonomy category used for classification.
     *
     * <p>Represents a label with an optional human-readable description.
     */
    @Node(label = "Category")
    @Getter
    @Setter
    @Builder
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Category extends Neo4jNode {

        /**
         * The unique name or identifier of the category.
         *
         * <p>Serialized as "name".
         */
        @JsonProperty("name")
        @JsonPropertyDescription("Name of the category")
        private String name;

        /**
         * A human-readable description explaining the meaning of the category.
         *
         * <p>Serialized as "description".
         */
        @JsonProperty("description")
        @JsonPropertyDescription("Description of the category")
        private String description;

        @JsonProperty("map-to")
        @JsonPropertyDescription("The label to map to")
        private String mapTo;

        @JsonProperty("examples")
        @JsonPropertyDescription("Example Questions and Answers for few-shot prompts")
        @Neo4jIgnore
        private List<TaxonomyExample> examples;

        @AfterDeserialization
        public void initialize() {
            Category existing = GlobalConfig.NEO4J_CLIENT.findNode(toPropertiesMap(), Category.class);

            if (existing == null) {
                saveWithExamples();
                return;
            }

            if (examplesNeedUpdate(existing)) {
                saveWithExamples();
            } else {
                setElementId(existing.getElementId());
            }
        }

        private boolean examplesNeedUpdate(Category existing) {
            if (examples == null || examples.isEmpty()) {
                return false;
            }

            List<TaxonomyExample> existingExamples = fetchExistingExamples(existing);

            if (existingExamples.size() != examples.size()) {
                return true;
            }

            Set<String> existingIds = existingExamples.stream()
                                                      .map(Neo4jNode::getElementId)
                                                      .collect(Collectors.toSet());

            Set<String> newIds = examples.stream()
                                         .map(Neo4jNode::getElementId)
                                         .collect(Collectors.toSet());

            return !existingIds.equals(newIds);
        }

        private List<TaxonomyExample> fetchExistingExamples(Category category) {
            String query = """
                    MATCH (category:%s)-[:%s]->(n:%s)
                    WHERE elementId(category) = $categoryId
                    RETURN n
                    """.formatted(
                    Neo4jNode.getLabel(Category.class),
                    Neo4jRelation.getType(HasExample.class),
                    Neo4jNode.getLabel(TaxonomyExample.class)
            );

            return GlobalConfig.NEO4J_CLIENT.executeQuery(
                    query,
                    Map.of("categoryId", category.getElementId()),
                    TaxonomyExample.class
            );
        }

        private void saveWithExamples() {
            GlobalConfig.NEO4J_CLIENT.saveNode(this);

            if (examples != null && !examples.isEmpty()) {
                examples.forEach(example -> {
                    HasExample relation = new HasExample();
                    relation.setStartNodeId(getElementId());
                    relation.setEndNodeId(example.getElementId());
                    GlobalConfig.NEO4J_CLIENT.createRelation(relation);
                });
            }
        }

        @Getter
        @Setter
        @Node(label = "TaxonomyExample")
        public static class TaxonomyExample extends Neo4jNode {
            @JsonProperty("question")
            private String question;

            @JsonProperty("answer")
            private String answer;

            @JsonProperty("explanation")
            private String explanation;

            @AfterDeserialization
            public void initialize() {
                TaxonomyExample existing = GlobalConfig.NEO4J_CLIENT.findNode(toPropertiesMap(), TaxonomyExample.class);
                if (existing != null) {
                    this.setElementId(existing.getElementId());
                } else {
                    GlobalConfig.NEO4J_CLIENT.saveNode(this);
                }
            }
        }
    }

    private void validateCategoriesMatch(Taxonomy existing) {
        List<Category> existingCategories = fetchExistingCategories(existing);

        if (categories == null || categories.isEmpty()) {
            if (!existingCategories.isEmpty()) {
                throw new IllegalStateException(
                        String.format(
                                "Taxonomy '%s' (version: %s) already exists with %d categories, but new instance has "
                                        + "no categories",
                                name, version, existingCategories.size())
                );
            }
            return;
        }

        if (existingCategories.size() != categories.size()) {
            throw new IllegalStateException(
                    String.format("Taxonomy '%s' (version: %s) category count mismatch: existing=%d, new=%d",
                                  name, version, existingCategories.size(), categories.size())
            );
        }

        Set<String> existingCategoryNames = existingCategories.stream()
                                                              .map(Category::getName)
                                                              .collect(Collectors.toSet());

        Set<String> newCategoryNames = categories.stream()
                                                 .map(Category::getName)
                                                 .collect(Collectors.toSet());

        if (!existingCategoryNames.equals(newCategoryNames)) {
            Set<String> missing = new HashSet<>(newCategoryNames);
            missing.removeAll(existingCategoryNames);

            Set<String> extra = new HashSet<>(existingCategoryNames);
            extra.removeAll(newCategoryNames);

            StringBuilder errorMsg = new StringBuilder(
                    String.format("Taxonomy '%s' (version: %s) category mismatch:\n", name, version)
            );

            if (!missing.isEmpty()) {
                errorMsg.append(String.format("  Missing in graph: %s\n", missing));
            }
            if (!extra.isEmpty()) {
                errorMsg.append(String.format("  Extra in graph: %s\n", extra));
            }

            throw new IllegalStateException(errorMsg.toString());
        }
    }

    private List<Category> fetchExistingCategories(Taxonomy taxonomy) {
        String query = """
                MATCH (taxonomy:%s)-[:%s]->(n:%s)
                WHERE elementId(taxonomy) = $taxonomyId
                RETURN n
                """.formatted(
                Neo4jNode.getLabel(Taxonomy.class),
                Neo4jRelation.getType(HasCategory.class),
                Neo4jNode.getLabel(Category.class)
        );

        return GlobalConfig.NEO4J_CLIENT.executeQuery(
                query,
                Map.of("taxonomyId", taxonomy.getElementId()),
                Category.class
        );
    }

    private void saveTaxonomyWithCategories() {
        GlobalConfig.NEO4J_CLIENT.saveNode(this);

        if (categories != null && !categories.isEmpty()) {
            categories.forEach(category -> {
                HasCategory relation = new HasCategory();
                relation.setStartNodeId(getElementId());
                relation.setEndNodeId(category.getElementId());
                GlobalConfig.NEO4J_CLIENT.createRelation(relation);
            });
        }
    }
}
