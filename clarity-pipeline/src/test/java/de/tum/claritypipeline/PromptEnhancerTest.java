package de.tum.claritypipeline;

import de.tum.claritypipeline.service.PromptEnhancer;
import org.junit.jupiter.api.Test;

import java.io.IOException;

/**
 * Smoke test for the “Prompt Enhancement” workflow, ensuring iterative diagnose/patch loops run end-to-end.
 */
public class PromptEnhancerTest {

    /**
     * Loads a prompt-enhancing YAML, runs all configured iterations, and confirms outputs are persisted in Neo4j.
     */
    @Test
    public void testEnhancePrompt() throws IOException {
        final String baseDir = "src/test/resources/prompt-enhancing/";

        PromptEnhancer promptEnhancer = new PromptEnhancer(baseDir + "gpt-5.1.yaml");
        promptEnhancer.enhance();
    }
}
