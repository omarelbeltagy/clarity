package de.tum.clarityneo4j.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.io.File;
import java.io.IOException;

/**
 * Class representing Neo4j exporter settings.
 */
@Getter
@Setter
public class Neo4jExporterConfig {

    @JsonProperty("batches")
    private Neo4jExporterBatchConfig batchConfig = new Neo4jExporterBatchConfig();
    /**
     * The credentials for the Neo4j database
     */
    @JsonProperty(value = "neo4j-credentials", index = 0)
    @JsonPropertyDescription("The neo4j credentials configuration.")
    @Neo4jIgnore
    @Setter(AccessLevel.NONE)
    private Neo4jCredentials neo4jCredentials = Neo4jCredentials.getDefault();

    public Neo4jExporterConfig() throws IOException {}

    /**
     * Load Neo4jExporterConfig from a YAML file.
     *
     * @param path the path to the YAML file
     * @return the loaded Neo4jCredentials
     * @throws IOException if an I/O error occurs
     */
    public static Neo4jExporterConfig load(String path) throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        return mapper.readValue(new File(path), Neo4jExporterConfig.class);
    }

    /**
     * Load Neo4jExporterConfig from the default YAML file located at "classpath:neo4j-exporter-config.yml".
     *
     * @return the loaded Neo4jExporterConfig
     * @throws IOException if an I/O error occurs
     */
    public static Neo4jExporterConfig getDefault() throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        try (var is = Neo4jCredentials.class.getClassLoader()
                                            .getResourceAsStream("neo4j-exporter-config.yaml")) {
            if (is == null) {
                throw new IOException("neo4j-exporter-config.yaml not found in classpath");
            }
            return mapper.readValue(is, Neo4jExporterConfig.class);
        }
    }

    @JsonSetter("neo4j-credentials")
    public void setNeo4jCredentials(Object raw) throws IOException {
        if (raw instanceof String s) {
            this.neo4jCredentials = Neo4jCredentials.load(s);
            return;
        }
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        this.neo4jCredentials = mapper.convertValue(raw, Neo4jCredentials.class);
    }

    @Getter
    @Setter
    public static class Neo4jExporterBatchConfig {
        @JsonProperty("delete")
        private int deleteBatchSize = 1000;
        @JsonProperty("import")
        private Neo4jExporterBatchConfigImport importConfig = new Neo4jExporterBatchConfigImport();
        @JsonProperty("export")
        private Neo4jExporterBatchConfigExport exportConfig = new Neo4jExporterBatchConfigExport();

        @Getter
        @Setter
        public static class Neo4jExporterBatchConfigImport {
            @JsonProperty("read")
            private int readBatchSize = 2000;
            @JsonProperty("relationships")
            private int relationshipBatchSize = 1000;
            @JsonProperty("nodes")
            private int nodeBatchSize = 1000;
        }

        @Getter
        @Setter
        public static class Neo4jExporterBatchConfigExport {
            @JsonProperty("relationships")
            private int relationshipBatchSize = 1000;
            @JsonProperty("nodes")
            private int nodeBatchSize = 1000;
        }
    }
}
