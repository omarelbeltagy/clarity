package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.claritypipeline.client.LocalClient;
import de.tum.claritypipeline.model.classification.ClassificationRequest;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.ModelProperties;
import de.tum.claritypipeline.utils.PromptUtils;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Classification strategy that simulates a multi-perspective discussion followed by a referee decision.
 * <p>
 * This strategy implements a two-phase classification approach:
 * <ol>
 *   <li><b>Discussion Phase</b>: For each taxonomy category, generate reasoning why the input
 *       might belong to that category</li>
 *   <li><b>Referee Phase</b>: Evaluate all category arguments and make a final classification decision</li>
 * </ol>
 *
 * <h2>Classification Process</h2>
 * <pre>
 * Input: Question & Answer
 *    ↓
 * Discussion Model (parallel for each category)
 *    → Category A: "Reasons why this is A..."
 *    → Category B: "Reasons why this is B..."
 *    → Category C: "Reasons why this is C..."
 *    ↓
 * Referee Model
 *    → Evaluates all arguments
 *    → Decides: Category B
 *    ↓
 * Output: Final Classification
 * </pre>
 *
 * <h2>Use Cases</h2>
 * <ul>
 *   <li>Complex classification tasks requiring multi-perspective analysis</li>
 *   <li>Scenarios where category boundaries are ambiguous</li>
 *   <li>When interpretability through explicit reasoning is important</li>
 * </ul>
 *
 * <h2>Requirements</h2>
 * <ul>
 *   <li>Discussion model must use JSON_OBJECT format (to capture reasoning per category)</li>
 *   <li>LocalClient is not supported (requires remote model API)</li>
 *   <li>Prompts must include {target_label} placeholder for discussion phase</li>
 *   <li>Prompts must include {reasons} placeholder for referee phase</li>
 * </ul>
 *
 * <h2>Performance Considerations</h2>
 * The discussion phase makes N parallel API calls (where N = number of taxonomy categories),
 * followed by one referee call. This makes it more expensive than single-model strategies.
 *
 * @see ClassificationResult
 */
@Node(label = "DiscussionStrategy")
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class DiscussionStrategy extends Neo4jNode implements ClassificationStrategy {
    @JsonIgnore
    private static final String TARGET_LABEL_PLACEHOLDER = "{target_label}";
    @JsonIgnore
    private static final String REASONS_PLACEHOLDER = "{reasons}";

    /**
     * Model configuration for generating category-specific reasoning.
     * This model is called once per taxonomy category to generate arguments
     * why the input might belong to that specific category.
     */
    @JsonProperty("discussion-model")
    @JsonPropertyDescription(
            "The model configuration for the models to come up with a reason for each taxonomy category.")
    private ModelProperties discussionModel;

    /**
     * Model configuration for making the final classification decision.
     * This model receives all category-specific reasonings and makes
     * an informed decision about the correct classification.
     */
    @JsonProperty("referee-model")
    @JsonPropertyDescription("The model configuration to use for the referee step.")
    private ModelProperties refereeModel;

    /**
     * Executes the discussion-based classification strategy.
     * <p>
     * Workflow:
     * <ol>
     *   <li>Validates configuration (JSON format, not LocalClient)</li>
     *   <li><b>Discussion Phase</b>: For each taxonomy category in parallel:
     *     <ul>
     *       <li>Builds prompt with target category</li>
     *       <li>Generates reasoning for that specific category</li>
     *       <li>Collects all reasonings</li>
     *     </ul>
     *   </li>
     *   <li><b>Referee Phase</b>:
     *     <ul>
     *       <li>Aggregates all category reasonings into one prompt</li>
     *       <li>Sends to referee model for final decision</li>
     *       <li>Returns classification result</li>
     *     </ul>
     *   </li>
     * </ol>
     *
     * @param request the classification request with question, answer, and taxonomy
     * @return classification result from the referee model
     * @throws UnsupportedOperationException if LocalClient is used or response format is not JSON_OBJECT
     */
    @Override
    public ClassificationResult execute(ClassificationRequest request) {
        if (discussionModel.getClient() instanceof LocalClient) {
            throw new UnsupportedOperationException("LocalClient is not supported for DiscussionStrategy");
        }
        if (discussionModel.getResponseFormat() != ModelProperties.ResponseFormat.JSON_OBJECT) {
            throw new UnsupportedOperationException(
                    "Only ResponseFormat.JSON_OBJECT is supported for the Discussion Model for the "
                            + "DiscussionStrategy, because the Referee Model "
                            + "needs to have an explanation from the Discussion Model available.");
        }

        String discussionPrompt = PromptUtils.replacePrompt(request, discussionModel,
                                                            ClassificationResult.JSON_SCHEME
        );
        List<ClassificationResult> discussions = new ArrayList<>();
        request.getTaxonomy().getCategories().parallelStream().forEach(category -> {
            String prompt = discussionPrompt.replace(TARGET_LABEL_PLACEHOLDER, category.getName());
            ClassificationResult result = discussionModel.getClient().makeRequest(prompt, ClassificationResult.class);
            if (result != null) {
                result.setName(category.getName());
                discussions.add(result);
            }
        });

        String refereePrompt = PromptUtils.replacePrompt(request,
                                                         refereeModel,
                                                         ClassificationResult.JSON_SCHEME
        ).replace(REASONS_PLACEHOLDER, buildReasonsForEachType(discussions));

        return refereeModel.getClient()
                           .makeRequest(refereePrompt, ClassificationResult.class);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T extends Neo4jNode> T getClassificationStrategyNode() {
        return (T) this;
    }

    /**
     * Aggregates all category-specific reasonings into a formatted string for the referee.
     * <p>
     * Format example:
     * <pre>
     * Reasons for *CategoryA* are: [reasoning for A]
     * Reasons for *CategoryB* are: [reasoning for B]
     * Reasons for *CategoryC* are: [reasoning for C]
     * </pre>
     *
     * @param discussions list of classification results containing category-specific explanations
     * @return formatted string with all reasonings
     */
    private String buildReasonsForEachType(List<ClassificationResult> discussions) {
        StringBuilder sb = new StringBuilder();
        for (ClassificationResult discussion : discussions) {
            sb.append("Reasons for *").append(discussion.getName()).append("* are: ");
            sb.append(discussion.getExplanation()).append("\n");
        }
        return sb.toString();
    }

}
