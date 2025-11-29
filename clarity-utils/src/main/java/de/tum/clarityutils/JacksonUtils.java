package de.tum.clarityutils;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class JacksonUtils {
    private JacksonUtils() {}

    public static <T> T readAndInit(
            com.fasterxml.jackson.databind.ObjectMapper mapper, java.io.File file, Class<T> clazz)
            throws java.io.IOException {
        T obj = mapper.readValue(file, clazz);
        callAfterDeserializationRecursive(obj, new HashSet<>());
        return obj;
    }

    public static <T> T convertAndInit(
            com.fasterxml.jackson.databind.ObjectMapper mapper, Object raw, Class<T> clazz) {
        T obj = mapper.convertValue(raw, clazz);
        callAfterDeserializationRecursive(obj, new HashSet<>());
        return obj;
    }

    private static void callAfterDeserializationRecursive(Object obj, Set<Object> visited) {
        if (obj == null || visited.contains(obj)) return;
        visited.add(obj);

        // ERST: Rekursiv in die Tiefe gehen
        for (Field field : obj.getClass().getDeclaredFields()) {
            // Ausnahme für Iterable/List - diese immer durchlaufen
            if (!Iterable.class.isAssignableFrom(field.getType()) && isJdkClass(field.getType())) {
                continue;
            }

            field.setAccessible(true);
            try {
                Object value = field.get(obj);
                if (value == null) continue;

                if (value instanceof Iterable<?> iterable) {
                    for (Object element : iterable) {
                        callAfterDeserializationRecursive(element, visited);
                    }
                } else if (value instanceof Map<?, ?> map) {
                    for (Object element : map.values()) {
                        callAfterDeserializationRecursive(element, visited);
                    }
                } else if (!isPrimitiveOrWrapper(value.getClass())) {
                    callAfterDeserializationRecursive(value, visited);
                }
            } catch (IllegalAccessException ignore) {
            }
        }

        // DANN: Methode auf diesem Objekt aufrufen
        for (Method method : obj.getClass().getDeclaredMethods()) {
            if (method.isAnnotationPresent(AfterDeserialization.class)) {
                method.setAccessible(true);
                try {
                    method.invoke(obj);
                } catch (Exception e) {
                    throw new IllegalStateException(
                            "Error invoking @AfterDeserialization on " + obj.getClass().getName(), e);
                }
            }
        }
    }

    private static boolean isJdkClass(Class<?> clazz) {
        return clazz.getPackageName().startsWith("java.")
                || clazz.getPackageName().startsWith("javax.")
                || clazz.getPackageName().startsWith("sun.");
    }

    private static boolean isPrimitiveOrWrapper(Class<?> type) {
        return type.isPrimitive()
                || type.equals(String.class)
                || Number.class.isAssignableFrom(type)
                || Boolean.class.equals(type)
                || Character.class.equals(type)
                || Enum.class.isAssignableFrom(type);
    }
}
