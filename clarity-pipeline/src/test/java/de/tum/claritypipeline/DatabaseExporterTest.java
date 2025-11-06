package de.tum.claritypipeline;

import de.tum.clarityneo4j.core.Neo4jExporter;
import org.junit.jupiter.api.Test;

import java.io.IOException;

/**
 * Tests database export and import utilities.
 *
 * <p>This class validates operations such as clearing the database, exporting the entire
 * graph to JSON, and importing a previously exported JSON file back into Neo4j.</p>
 */
public class DatabaseExporterTest {
    /**
     * Neo4j exporter utility used to perform export/import operations.
     */
    private final Neo4jExporter neo4jExporter = new Neo4jExporter();

    /**
     * Default constructor initializes resources required for tests.
     *
     * @throws IOException if exporter initialization fails
     */
    public DatabaseExporterTest() throws IOException {}

    /**
     * Delete all nodes and relationships from the Neo4j database.
     *
     * <p>This test ensures that the clear operation completes without throwing exceptions.
     * It is intended to leave the database in a clean state before other tests run.</p>
     */
    @Test
    public void testClearDatabase() {
        neo4jExporter.clearDatabase();
    }

    /**
     * Export the Neo4j database to a JSON file.
     *
     * <p>Saves all nodes and relationships including their properties to the given output file.
     * The produced JSON can be later used for import or analysis.</p>
     *
     * @throws IOException if writing the export file fails
     */
    @Test
    public void testExportDatabase() throws IOException {
        neo4jExporter.exportAsJson("src/test/resources/neo4j-export/11_04_2025_INITIAL_EXPERIMENTS.json");
    }

    /**
     * Import a Neo4j database from a JSON file.
     *
     * <p>Loads nodes and relationships (with properties) from the specified JSON file into Neo4j.
     * Use this to restore a saved database state for testing.</p>
     *
     * @throws IOException if reading the import file fails
     */
    @Test
    public void testImportDatabase() throws IOException {
        neo4jExporter.importFromJson("src/test/resources/neo4j-export/11_04_2025_INITIAL_EXPERIMENTS.json");
    }
}
