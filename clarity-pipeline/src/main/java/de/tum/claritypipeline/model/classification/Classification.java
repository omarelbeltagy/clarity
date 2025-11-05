package de.tum.claritypipeline.model.classification;

import com.fasterxml.jackson.annotation.JsonIgnore;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.claritypipeline.model.config.ClassificationProperties;
import de.tum.claritypipeline.model.relation.HasRun;
import lombok.*;

import java.util.List;

/**
 * Represents a classification grouping entity stored in Neo4j.
 *
 * <p>Used to link classification results and metadata to a named run.
 */
@Node(label = "Classification")
@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class Classification extends Neo4jNode {

    /**
     * The name of the classification.
     *
     * <p>Used as an identifier when creating run nodes in Neo4j.
     */
    private String name;

    /**
     * The classification runs connected to this classification.
     */
    @JsonIgnore
    @Neo4jIgnore
    private List<ClassificationProperties> children;

    /**
     * Retrieves the child classification properties connected to this classification via "HAS_RUN" relationships.
     *
     * @param neo4jClient The Neo4j client used to execute the query.
     * @return A list of ClassificationProperties nodes connected to this classification.
     */
    public List<ClassificationProperties> getChildren(Neo4jClient neo4jClient) {
        if (this.children != null) {
            return this.children;
        } else {
            if (getElementId() == null) {
                return List.of();
            } else {
                String query = String.format("""
                                                     MATCH (n:%s)<-[:%s]-(u:%s)
                                                     WHERE elementId(u) = '%s'
                                                     RETURN n
                                                     """,
                                             Neo4jNode.getLabel(ClassificationProperties.class),
                                             Neo4jRelation.getType(HasRun.class),
                                             Neo4jNode.getLabel(Classification.class),
                                             getElementId()
                );
                List<ClassificationProperties> children = neo4jClient.executeQuery(query,
                                                                                   ClassificationProperties.class);
                this.children = children;
                return children;
            }
        }
    }
}
