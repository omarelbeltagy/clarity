package de.tum.claritypipeline;

import de.tum.clarityneo4j.core.Neo4jExporter;
import org.junit.jupiter.api.Test;

import java.io.IOException;

/**
 * Regression tests for the README’s “Data Management” responsibilities.
 * <p>
 * Covers the life cycle of wiping the Neo4j graph, exporting a full snapshot, and restoring it again,
 * ensuring reproducible experiments across machines.
 * </p>
 */
public class DatabaseExporterTest {
    /**
     * Neo4j exporter utility used to perform export/import operations.
     */
    private final Neo4jExporter neo4jExporter = new Neo4jExporter();
    /*
    private final Neo4jExporter neo4jExporter = new Neo4jExporter(Neo4jExporterConfig.load(
            "src/test/resources/neo4j-exporter-config-reduced-batch-size.yaml"));
     */


    /**
     * Default constructor initializes resources required for tests.
     *
     * @throws IOException if exporter initialization fails
     */
    public DatabaseExporterTest() throws IOException {}

    /**
     * Ensures the exporter can delete every QA/taxonomy/strategy node so the next pipeline run starts from a clean
     * slate.
     */
    @Test
    public void testClearDatabase() {
        neo4jExporter.clearDatabase();
    }

    /**
     * Validates that the entire ontology (datasets, strategies, evaluations) can be serialized to JSON for backup or
     * sharing.
     */
    @Test
    public void testExportDatabase() throws IOException {
        neo4jExporter.exportAsJson("src/test/resources/neo4j-export/12_21_2025.json", false);
    }

    /**
     * Confirms a previously exported snapshot can be re-imported to rebuild the pipeline state described in the README.
     */
    @Test
    public void testImportDatabase() throws IOException {
        neo4jExporter.clearDatabase();
        neo4jExporter.importFromJson("src/test/resources/neo4j-export/12_21_2025.json");
    }
}
