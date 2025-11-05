package de.tum.clarityutils;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.victools.jsonschema.generator.OptionPreset;
import com.github.victools.jsonschema.generator.SchemaGenerator;
import com.github.victools.jsonschema.generator.SchemaGeneratorConfigBuilder;
import com.github.victools.jsonschema.generator.SchemaVersion;
import com.github.victools.jsonschema.module.jackson.JacksonModule;
import lombok.AccessLevel;
import lombok.Getter;

/**
 * JsonScheme is a small utility class that generates a JSON Schema for a given Java type
 * using the victools JSON Schema generator and exposes convenient methods to obtain
 * the generated schema as pretty-printed JSON strings.
 * <p>
 * Usage:
 * - Instantiate with the target class: new JsonScheme(MyClass.class)
 * - Call toJson() to obtain the full schema as a formatted JSON string.
 * - Call getPropertiesString() to obtain the "properties" node of the schema as formatted JSON.
 * <p>
 * Implementation notes:
 * - The class uses victools' SchemaGenerator configured for DRAFT_2020_12 and Jackson module support.
 * - Jackson's ObjectMapper is used to convert the generated JsonNode into pretty JSON strings.
 * - JSON processing errors are caught and result in a fallback empty JSON object string "{}".
 *
 * @param <T> the Java type for which the JSON Schema is generated
 */
@Getter
public class JsonScheme<T> {
    @Getter(AccessLevel.NONE)
    private final Class<T> type;
    @Getter(AccessLevel.NONE)
    private final ObjectMapper mapper;
    private final JsonNode scheme;

    /**
     * Construct a JsonScheme for the provided Java type.
     * <p>
     * The constructor initializes the victools SchemaGenerator with a default configuration:
     * - SchemaVersion.DRAFT_2020_12
     * - OptionPreset.PLAIN_JSON
     * - JacksonModule enabled so Jackson-specific types are handled
     * <p>
     * It builds the schema once during construction and stores it in the 'scheme' field.
     *
     * @param type the Class object representing the target Java type
     */
    public JsonScheme(Class<T> type) {
        this.type = type;
        SchemaGeneratorConfigBuilder configBuilder = new SchemaGeneratorConfigBuilder(
                SchemaVersion.DRAFT_2020_12, OptionPreset.PLAIN_JSON)
                .with(new JacksonModule());
        SchemaGenerator generator = new SchemaGenerator(configBuilder.build());

        this.mapper = new ObjectMapper();
        this.scheme = generator.generateSchema(type);
    }

    /**
     * Return the generated JSON Schema as a pretty-printed JSON string.
     * <p>
     * If an error occurs during JSON serialization (which is unlikely since the input is a Jackson JsonNode),
     * the method catches JsonProcessingException and returns the fallback string "{}".
     *
     * @return the full JSON Schema as a formatted JSON string, or "{}" on serialization failure
     */
    public String toJson() {
        try {
            return mapper.writerWithDefaultPrettyPrinter().writeValueAsString(scheme);
        } catch (JsonProcessingException e) {
            return "{}";
        }
    }

    /**
     * Return the "properties" section of the generated JSON Schema as a pretty-printed JSON string.
     * <p>
     * This is useful when only the schema properties (the declared fields and their types) are needed,
     * rather than the entire schema document. On serialization failure the method returns "{}".
     *
     * @return the "properties" node of the schema as a formatted JSON string, or "{}" on serialization failure
     */
    public String getPropertiesString() {
        try {
            return mapper.writerWithDefaultPrettyPrinter().writeValueAsString(scheme.get("properties"));
        } catch (JsonProcessingException e) {
            return "{}";
        }
    }
}
