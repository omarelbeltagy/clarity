package de.tum.claritypipeline.model.core;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import lombok.*;

import java.util.List;

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
public class Category extends Neo4jNode {

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
    }
}
