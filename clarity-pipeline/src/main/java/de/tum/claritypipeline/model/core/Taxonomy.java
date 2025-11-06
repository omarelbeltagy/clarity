package de.tum.claritypipeline.model.core;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import lombok.*;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.List;

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
}
