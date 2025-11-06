package de.tum.clarityneo4j.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import lombok.*;

import java.io.File;
import java.io.IOException;

/**
 * Class representing Neo4j database credentials.
 */
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@Builder
public class Neo4jCredentials {
    /**
     * Username for Neo4j database
     */
    @JsonProperty("neo4j-user")
    @JsonPropertyDescription("Username for Neo4j database")
    private String neo4jUser;

    /**
     * Password for Neo4j database
     */
    @JsonProperty("neo4j-password")
    @JsonPropertyDescription("Password for Neo4j database")
    private String neo4jPassword;

    /**
     * URL for Neo4j database
     */
    @JsonProperty("neo4j-url")
    @JsonPropertyDescription("URL for Neo4j database. Example: bolt://localhost:7687")
    private String neo4jUrl;

    /**
     * Load Neo4jCredentials from a YAML file.
     *
     * @param path the path to the YAML file
     * @return the loaded Neo4jCredentials
     * @throws IOException if an I/O error occurs
     */
    public static Neo4jCredentials load(String path) throws IOException {
        if (path == null || path.isEmpty()) {
            throw new IllegalArgumentException("Path to neo4j-credentials.yaml must be provided");
        }
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        return mapper.readValue(new File(path), Neo4jCredentials.class);
    }

    /**
     * Load Neo4jCredentials from the default YAML file located at "classpath:neo4j-credentials.yml".
     *
     * @return the loaded Neo4jCredentials
     * @throws IOException if an I/O error occurs
     */
    public static Neo4jCredentials getDefault() throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        try (var is = Neo4jCredentials.class.getClassLoader()
                                            .getResourceAsStream("neo4j-credentials.yaml")) {
            if (is == null) {
                throw new IOException("neo4j-credentials.yaml not found in classpath");
            }
            return mapper.readValue(is, Neo4jCredentials.class);
        }
    }
}
