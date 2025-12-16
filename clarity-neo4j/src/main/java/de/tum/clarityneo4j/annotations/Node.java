package de.tum.clarityneo4j.annotations;

import com.fasterxml.jackson.annotation.JsonIgnore;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation to specify the Neo4j node label for a class.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Node {
    @JsonIgnore
    String label() default "";
}