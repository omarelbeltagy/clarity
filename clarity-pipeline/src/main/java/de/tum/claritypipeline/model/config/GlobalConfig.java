package de.tum.claritypipeline.model.config;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.model.Neo4jCredentials;

import java.io.IOException;

/**
 * Holds the lazily shared Neo4j credentials/client used by all model/config objects.
 * <p>Initialized once so that every {@code @AfterDeserialization} hook can persist/find nodes consistently.</p>
 */
public class GlobalConfig {
    public static Neo4jCredentials NEO4J_CREDENTIALS;
    public static Neo4jClient NEO4J_CLIENT;

    static {
        try {
            NEO4J_CREDENTIALS = Neo4jCredentials.getDefault();
            NEO4J_CLIENT = new Neo4jClient(NEO4J_CREDENTIALS);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
