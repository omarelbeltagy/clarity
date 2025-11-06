package de.tum.clarityneo4j.core;

import com.fasterxml.jackson.annotation.JsonIgnore;
import de.tum.clarityneo4j.annotations.Neo4jIgnore;
import de.tum.clarityneo4j.annotations.Neo4jProperty;
import lombok.Getter;
import lombok.Setter;
import org.neo4j.driver.Value;
import org.neo4j.driver.internal.value.NodeValue;
import org.neo4j.driver.types.Node;

import java.io.Serializable;
import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Abstract base class for Neo4j nodes.
 * Provides reflection-based mapping between Java fields and Neo4j node properties.
 * All Cypher generation uses parameterized queries to prevent injection attacks.
 */
@Getter
@Setter
public abstract class Neo4jNode implements Serializable {

    private static final Map<Class<?>, List<Field>> FIELD_CACHE = new ConcurrentHashMap<>();
    private static final Map<Class<?>, String> LABEL_CACHE = new ConcurrentHashMap<>();

    @Neo4jIgnore
    @JsonIgnore
    private String elementId;

    /**
     * Returns the Neo4j label for the given class.
     */
    public static <T extends Neo4jNode> String getLabel(Class<T> clazz) {
        return LABEL_CACHE.computeIfAbsent(clazz, cls -> {
            de.tum.clarityneo4j.annotations.Node ann = cls.getAnnotation(de.tum.clarityneo4j.annotations.Node.class);
            if (ann == null) return cls.getSimpleName();
            return ann.label().isEmpty() ? cls.getSimpleName() : ann.label();
        });
    }

    /**
     * Creates an instance from a Neo4j NodeValue.
     */
    public static <T extends Neo4jNode> T fromNodeValue(NodeValue c, Class<T> clazz) {
        try {
            T instance = clazz.getDeclaredConstructor().newInstance();
            Node node = c.asNode();

            for (Field f : getMappableFields(clazz)) {
                String key = resolvePropertyName(f);
                if (node.containsKey(key)) {
                    Object v = convertValue(node.get(key), f.getType());
                    setFieldValueQuiet(instance, f, v);
                }
            }
            instance.setElementId(node.elementId());
            return instance;
        } catch (Exception e) {
            throw new IllegalStateException("fromNodeValue failed for " + clazz.getName(), e);
        }
    }

    /**
     * Returns list of fields that should be mapped to Neo4j properties.
     */
    private static List<Field> getMappableFields(Class<?> clazz) {
        return FIELD_CACHE.computeIfAbsent(clazz, c -> {
            List<Field> all = new ArrayList<>();
            Class<?> k = c;
            while (k != null && k != Object.class) {
                for (Field f : k.getDeclaredFields()) {
                    if (isMappable(f)) all.add(f);
                }
                k = k.getSuperclass();
            }
            all.sort(Comparator.comparing(Field::getName));
            return Collections.unmodifiableList(all);
        });
    }

    private static boolean isMappable(Field f) {
        int mod = f.getModifiers();
        if (java.lang.reflect.Modifier.isStatic(mod)) return false;
        if (java.lang.reflect.Modifier.isTransient(mod)) return false;
        return !f.isAnnotationPresent(Neo4jIgnore.class);
    }

    private static String resolvePropertyName(Field f) {
        Neo4jProperty ann = f.getAnnotation(Neo4jProperty.class);
        return ann != null ? ann.value() : f.getName();
    }

    private static Object convertValue(Value neo4jValue, Class<?> target) {
        if (neo4jValue == null || neo4jValue.isNull()) return null;

        if (target == String.class) return neo4jValue.asString(null);
        if (target == Integer.class || target == int.class) return neo4jValue.asInt();
        if (target == Long.class || target == long.class) return neo4jValue.asLong();
        if (target == Double.class || target == double.class) return neo4jValue.asDouble();
        if (target == Boolean.class || target == boolean.class) return neo4jValue.asBoolean();

        if (target.isArray()) {
            Class<?> componentType = target.getComponentType();
            List<Object> list = neo4jValue.asList(v -> v.isNull() ? null : convertValue(v, componentType));
            Object array = java.lang.reflect.Array.newInstance(componentType, list.size());
            for (int i = 0; i < list.size(); i++) {
                java.lang.reflect.Array.set(array, i, list.get(i));
            }
            return array;
        }

        if (List.class.isAssignableFrom(target)) {
            return neo4jValue.asList(v -> v.isNull() ? null : v.asObject());
        }

        return neo4jValue.asObject();
    }

    private static Object getFieldValueQuiet(Object obj, Field f) {
        try {
            f.setAccessible(true);
            return f.get(obj);
        } catch (Exception e) {
            throw new IllegalStateException("Read field failed " + f, e);
        }
    }

    private static void setFieldValueQuiet(Object obj, Field f, Object value) {
        try {
            f.setAccessible(true);
            f.set(obj, value);
        } catch (Exception e) {
            throw new IllegalStateException("Write field failed " + f, e);
        }
    }

    public String getLabel() {
        return getLabel(getClass());
    }

    /**
     * Returns properties as a Map suitable for use as Cypher parameters.
     * This is the SAFE way to pass properties to Neo4j.
     */
    public Map<String, Object> toPropertiesMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        for (Field f : getMappableFields(getClass())) {
            Object val = getFieldValueQuiet(this, f);
            if (val != null) {
                String key = resolvePropertyName(f);
                map.put(key, val);
            }
        }
        return map;
    }

    /**
     * DEPRECATED: Use toPropertiesMap() with parameterized queries instead.
     * This method creates inline Cypher which can be unsafe.
     *
     * @deprecated Use {@link #toPropertiesMap()} with parameterized queries
     */
    @Deprecated
    public String toNeo4jCypherMap() {
        return toCypherMap(toPropertiesMap());
    }

    /**
     * Returns Cypher node pattern with parameter placeholder.
     * Example: (n:User $props)
     * Use with: Map.of("props", node.toPropertiesMap())
     */
    public String toNeo4jPattern(String alias) {
        String aliasStr = alias == null || alias.isEmpty() ? "" : alias;
        return String.format("(%s:%s $%sprops)", aliasStr, getLabel(), aliasStr);
    }

    /**
     * DEPRECATED: Use toNeo4jPattern() with parameterized queries instead.
     *
     * @deprecated Use {@link #toNeo4jPattern(String)} with parameterized queries
     */
    @Deprecated
    public String toNeo4j(String alias) {
        return buildNeo4jNode(toPropertiesMap(), alias);
    }

    /**
     * DEPRECATED: Use toNeo4jPattern() with parameterized queries instead.
     *
     * @deprecated Use {@link #toNeo4jPattern(String)} with parameterized queries
     */
    @Deprecated
    public String toNeo4j() {
        return toNeo4j("");
    }

    /**
     * DEPRECATED: Builds inline Cypher which can be unsafe.
     */
    @Deprecated
    protected String buildNeo4jNode(Map<String, Object> props, String alias) {
        StringBuilder sb = new StringBuilder();
        sb.append("(").append(alias).append(":").append(getLabel()).append(" ");
        sb.append(toCypherMap(props));
        sb.append(")");
        return sb.toString();
    }

    /**
     * DEPRECATED: Creates inline Cypher map.
     * For backwards compatibility only - use parameterized queries instead.
     */
    @Deprecated
    protected String toCypherMap(Map<String, Object> props) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        boolean first = true;
        for (Map.Entry<String, Object> e : props.entrySet()) {
            String cypherLiteral = toCypherLiteral(e.getValue());
            if (cypherLiteral.isEmpty()) continue;
            if (!first) sb.append(", ");
            first = false;
            // Escape property name
            String escapedKey = escapeIdentifier(e.getKey());
            sb.append(escapedKey).append(": ").append(cypherLiteral);
        }
        sb.append("}");
        return sb.toString();
    }

    /**
     * Escapes a Cypher identifier (property name, label, etc.).
     * Wraps in backticks if necessary.
     */
    private String escapeIdentifier(String identifier) {
        // If contains special characters, wrap in backticks
        if (identifier.matches("^[a-zA-Z_][a-zA-Z0-9_]*$")) {
            return identifier;
        }
        return "`" + identifier.replace("`", "``") + "`";
    }

    /**
     * DEPRECATED: Converts value to inline Cypher literal.
     * This is unsafe for user input - use parameterized queries instead.
     */
    @Deprecated
    protected String toCypherLiteral(Object v) {
        if (v == null) return "";
        if (v instanceof Number || v instanceof Boolean) return String.valueOf(v);
        if (v instanceof Collection<?> col) {
            StringBuilder sb = new StringBuilder("[");
            boolean first = true;
            for (Object x : col) {
                if (!first) sb.append(", ");
                first = false;
                sb.append(toCypherLiteral(x));
            }
            sb.append("]");
            return sb.toString();
        }
        if (v.getClass().isArray()) {
            StringBuilder sb = new StringBuilder("[");
            int length = java.lang.reflect.Array.getLength(v);
            for (int i = 0; i < length; i++) {
                if (i > 0) sb.append(", ");
                Object x = java.lang.reflect.Array.get(v, i);
                sb.append(toCypherLiteral(x));
            }
            sb.append("]");
            return sb.toString();
        }
        String s = v.toString().replace("\\", "\\\\").replace("'", "\\'");
        if (s.isEmpty() || s.equals("null")) {
            return "";
        }
        return "'" + s + "'";
    }
}