package de.tum.claritypipeline;

import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.claritypipeline.service.EmbeddingService;
import org.junit.jupiter.api.Test;

import java.io.IOException;

/**
 * Verifies the embedding subsystem mentioned in the README (“RAG Support”) can populate Neo4j indices.
 */
public class EmbeddingServiceTest {
    public EmbeddingServiceTest() {}

    /**
     * Boots the embedding service and generates vectors for all QA nodes so downstream RAG-enabled models can
     * retrieve examples.
     */
    @Test
    public void testGenerateEmbeddingsForQAs() throws IOException {
        EmbeddingService.initialize(Neo4jCredentials.getDefault(),
                                    "text-embedding-3-large");
        final EmbeddingService embeddingService = EmbeddingService.getInstance();
        String query = """
                MATCH (n:QA)
                RETURN n
                """;
        embeddingService.generateEmbeddingsForQAs(query);
    }
}
