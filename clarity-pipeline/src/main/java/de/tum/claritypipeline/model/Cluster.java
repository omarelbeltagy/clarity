package de.tum.claritypipeline.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Node;
import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.claritypipeline.model.relation.BelongsTo;
import lombok.*;

import java.util.List;

@Node(label = "Cluster")
@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class Cluster extends Neo4jNode {

    private String name;

    @JsonIgnore
    @Neo4jIgnore
    private List<Classification> children;

    public List<Classification> getChildren(Neo4jClient neo4jClient) {
        if (this.children != null) {
            return this.children;
        } else {
            if (getElementId() == null) {
                return List.of();
            } else {
                String query = String.format("""
                                                     MATCH (n:%s)-[:%s]->(u:%s)
                                                     WHERE elementId(u) = '%s'
                                                     RETURN n
                                                     """,
                                             Neo4jNode.getLabel(Classification.class),
                                             Neo4jRelation.getType(BelongsTo.class),
                                             Neo4jNode.getLabel(Cluster.class),
                                             getElementId()
                );
                List<Classification> children = neo4jClient.executeQuery(query, Classification.class);
                this.children = children;
                return children;
            }
        }
    }
}
