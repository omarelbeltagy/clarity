package de.tum.claritypipeline.model.relation;

import de.tum.clarityneo4j.annotations.Relation;
import de.tum.clarityneo4j.core.Neo4jRelation;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Relation(type = "HAS_PATTERN_PROPERTIES", direction = Relation.Direction.OUTGOING)
@Getter
@Setter
@Builder
@AllArgsConstructor
public class HasPatternProperties extends Neo4jRelation {}
