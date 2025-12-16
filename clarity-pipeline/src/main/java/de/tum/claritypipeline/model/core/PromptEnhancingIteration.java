package de.tum.claritypipeline.model.core;

import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import lombok.*;

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