package de.tum.claritypipeline;

import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.claritypipeline.service.EmbeddingService;
import org.junit.jupiter.api.Test;

import java.io.IOException;

public class EmbeddingServiceTest {
    public EmbeddingServiceTest() {}

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
