package de.tum.clarityutils;

import com.google.gson.Gson;

import java.util.Arrays;
import java.util.List;

/**
 * Utility class for serializing and deserializing objects to and from JSON using Gson.
 * <p>
 * This class provides static methods to serialize Java objects to JSON strings and to deserialize
 * JSON strings back into Java objects or lists of objects. It is designed to simplify the conversion
 * between Java objects and their JSON representations, supporting both single objects and arrays/lists.
 * <p>
 * Example usage:
 * <pre>
 *     String json = SerializationUtils.serialize(myObject);
 *     MyClass obj = SerializationUtils.deserialize(json, MyClass.class);
 *     List&lt;MyClass&gt; list = SerializationUtils.deserializeList(jsonArray, MyClass[].class);
 * </pre>
 */
public class SerializationUtils {

    /**
     * Serializes an object to its JSON string representation.
     * <p>
     * Uses Gson to convert the provided object into a JSON string. If the object is {@code null},
     * the method returns {@code null}.
     *
     * @param obj the object to serialize
     * @return the JSON string representation of the object, or {@code null} if the object is {@code null}
     */
    public static String serialize(Object obj) {
        if (obj == null) {
            return null;
        }
        Gson gson = new Gson();
        return gson.toJson(obj);
    }

    /**
     * Deserializes a JSON string into an object of the specified class.
     * <p>
     * Uses Gson to parse the JSON string and create an instance of the specified class.
     * If the JSON string is {@code null}, the method returns {@code null}.
     *
     * @param json        the JSON string
     * @param targetClass the class of T
     * @param <T>         the type of the desired object
     * @return the deserialized object, or {@code null} if the JSON string is {@code null}
     */
    public static <T> T deserialize(String json, Class<T> targetClass) {
        if (json == null) {
            return null;
        }
        Gson gson = new Gson();
        return gson.fromJson(json, targetClass);
    }

    /**
     * Deserializes a JSON string into a list of objects of the specified class.
     * <p>
     * Uses Gson to parse the JSON string and create an array of the specified type, then converts it to a list.
     * If the JSON string is {@code null}, an empty list is returned.
     *
     * @param json        the JSON string
     * @param targetClass the array class of T (e.g., MyClass[].class)
     * @param <T>         the type of the desired objects
     * @return a list of deserialized objects, or an empty list if the JSON string is {@code null}
     */
    public static <T> List<T> deserializeList(String json, Class<T[]> targetClass) {
        if (json == null || json.isEmpty() || json.equals("null")) {
            json = "[]"; // Return an empty list if json is null
        }
        Gson gson = new Gson();
        T[] array = gson.fromJson(json, targetClass);
        return Arrays.asList(array);
    }
}
