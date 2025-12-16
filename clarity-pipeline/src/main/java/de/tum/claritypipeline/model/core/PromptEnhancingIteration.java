package de.tum.claritypipeline.model.core;

import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import lombok.*;

/**
 * Represents a single iteration of prompt enhancement in the Clarity pipeline.
 *
 * <p>
 * This node captures the details of each iteration where a prompt is diagnosed,
 * patched, and revised to improve its effectiveness. It includes information
 * about the requests and results of diagnosing and patching, as well as the
 * initial and enhanced prompts.
 * </p>
 */
@Node(label = "PromptEnhancingIteration")
@Getter
@Setter
@Builder()
@AllArgsConstructor
@NoArgsConstructor
public class PromptEnhancingIteration extends Neo4jNode {
    private Integer iterationNumber;

    private String diagnoseRequest;

    private String diagnoseResult;

    private String patchRequest;

    private String patchResult;

    private String revisedPrompt;

    private String revisedTaxonomy;

    private String initialPrompt;

    private String enhancedPrompt;

    private String failureModesResult;
}