package de.tum.clarityutils;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.TypeFactory;

import java.util.Collections;
import java.util.List;

/**
 * Utility class for serializing and deserializing objects to and from JSON using Jackson.
 */
public class SerializationUtils {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * Serializes an object to its JSON string representation.
     *
     * @param obj the object to serialize
     * @return the JSON string representation of the object, or {@code null} if the object is {@code null}
     */
    public static String serialize(Object obj) {
        if (obj == null) {
            return null;
        }
        try {
            return MAPPER.writeValueAsString(obj);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to serialize object of type " + obj.getClass(), e);
        }
    }

    /**
     * Deserializes a JSON string into an object of the specified class.
     *
     * @param json        the JSON string
     * @param targetClass the class of T
     * @param <T>         the type of the desired object
     * @return the deserialized object, or {@code null} if the JSON string is {@code null}
     */
    public static <T> T deserialize(String json, Class<T> targetClass) {
        if (json == null || json.isEmpty() || json.equals("null")) {
            return null;
        }
        try {
            return MAPPER.readValue(json, targetClass);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to deserialize JSON to " + targetClass.getSimpleName(), e);
        }
    }

    /**
     * Deserializes a JSON string into a list of objects of the specified class.
     *
     * @param json        the JSON string
     * @param targetClass the class of T
     * @param <T>         the type of the desired objects
     * @return a list of deserialized objects, or an empty list if the JSON string is {@code null}
     */
    public static <T> List<T> deserializeList(String json, Class<T> targetClass) {
        if (json == null || json.isEmpty() || json.equals("null")) {
            return Collections.emptyList();
        }
        try {
            return MAPPER.readValue(
                    json,
                    TypeFactory.defaultInstance().constructCollectionType(List.class, targetClass)
            );
        } catch (JsonProcessingException e) {
            throw new IllegalStateException(
                    "Failed to deserialize JSON array to List<" + targetClass.getSimpleName() + ">", e);
        }
    }
}
