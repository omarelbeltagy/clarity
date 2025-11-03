package de.tum.clarityneo4j.annotations;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation to specify the Neo4j relationship type and direction for a class.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Relation {
    String type();

    Direction direction() default Direction.OUTGOING;

    enum Direction {
        OUTGOING, INCOMING, UNDIRECTED
    }
}