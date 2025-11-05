package de.tum.clarityneo4j.model;

import de.tum.clarityneo4j.core.Neo4jNode;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Neo4jEmbeddingSearchResult<T extends Neo4jNode> {
    private T node;
    private double score;
}
