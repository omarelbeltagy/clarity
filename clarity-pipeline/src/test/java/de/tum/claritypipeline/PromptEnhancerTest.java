package de.tum.claritypipeline;

import de.tum.claritypipeline.service.PromptEnhancer;
import org.junit.jupiter.api.Test;

import java.io.IOException;

public class PromptEnhancerTest {

    @Test
    public void testEnhancePrompt() throws IOException {
        final String baseDir = "src/test/resources/prompt-enhancing/";

        PromptEnhancer promptEnhancer = new PromptEnhancer(baseDir + "gpt-5.1.yaml");
        promptEnhancer.enhance();
    }
}
