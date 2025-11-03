package de.tum.claritypipeline.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import lombok.*;

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
}
