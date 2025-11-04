package de.tum.claritypipeline.strategy;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import de.tum.claritypipeline.model.ClassificationRequest;
import de.tum.claritypipeline.model.ClassificationResult;

@JsonTypeInfo(
        use = JsonTypeInfo.Id.NAME,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type"
)
@JsonSubTypes({
        @JsonSubTypes.Type(value = SingleStrategy.class, name = "single"),
})
public interface ClassificationStrategy {
    ClassificationResult execute(ClassificationRequest request);
}
