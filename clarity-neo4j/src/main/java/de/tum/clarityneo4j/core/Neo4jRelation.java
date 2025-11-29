package de.tum.clarityneo4j.core;

import de.tum.clarityneo4j.annotations.Neo4jProperty;
import de.tum.clarityneo4j.annotations.Relation;
import lombok.Getter;
import lombok.Setter;
import org.neo4j.driver.Value;
import org.neo4j.driver.internal.value.RelationshipValue;
import org.neo4j.driver.types.Relationship;

import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Abstract base class for Neo4j relationships.
 * Provides reflection-based mapping between Java fields and Neo4j relationship properties.
 * All Cypher generation uses parameterized queries to prevent injection attacks.
 */
@Getter
@Setter
public abstract class Neo4jRelation {

    /**
     * Cache for mappable fields per class to improve reflection performance.
     */
    private static final Map<Class<?>, List<Field>> FIELD_CACHE = new ConcurrentHashMap<>();

    /**
     * Neo4j element ID for this relationship.
     */
    private String elementId;

    /**
     * Element ID of the start node.
     */
    private String startNodeId;

    /**
     * Element ID of the end node.
     */
    private String endNodeId;

    /**
     * Returns the Neo4j relationship type for the given class.
     */
    public static <T extends Neo4jRelation> String getType(Class<T> clazz) {
        Relation ann = clazz.getAnnotation(Relation.class);
        return ann != null ? ann.type() : clazz.getSimpleName();
    }

    /**
     * Creates an instance from a Neo4j RelationshipValue.
     */
    public static <T extends Neo4jRelation> T fromRelationValue(RelationshipValue rv, Class<T> clazz) {
        try {
            T instance = clazz.getDeclaredConstructor().newInstance();
            Relationship rel = rv.asRelationship();

            instance.setElementId(rel.elementId());
            instance.setStartNodeId(rel.startNodeElementId());
            instance.setEndNodeId(rel.endNodeElementId());

            for (Field f : getMappableFields(clazz)) {
                String key = resolvePropertyName(f);
                if (rel.containsKey(key)) {
                    Object val = convertValue(rel.get(key), f.getType());
                    setFieldValue(instance, f, val);
                }
            }
            return instance;
        } catch (Exception e) {
            throw new IllegalStateException("fromRelationValue failed for " + clazz.getName(), e);
        }
    }

    /**
     * Returns list of fields that should be mapped to Neo4j properties.
     */
    private static List<Field> getMappableFields(Class<?> clazz) {
        return FIELD_CACHE.computeIfAbsent(clazz, c -> {
            List<Field> fields = new ArrayList<>();
            for (Field f : c.getDeclaredFields()) {
                if (!f.isSynthetic() && !isSystemField(f)) {
                    fields.add(f);
                }
            }
            fields.sort(Comparator.comparing(Field::getName));
            return Collections.unmodifiableList(fields);
        });
    }

    /**
     * Checks if field is a system field (elementId, startNodeId, endNodeId).
     */
    private static boolean isSystemField(Field f) {
        String name = f.getName();
        return "elementId".equals(name) || "startNodeId".equals(name) || "endNodeId".equals(name);
    }

    /**
     * Converts a Neo4j Value to the target Java type.
     */
    private static Object convertValue(Value val, Class<?> target) {
        if (val == null || val.isNull()) return null;
        if (target == String.class) return val.asString();
        if (target == int.class || target == Integer.class) return val.asInt();
        if (target == long.class || target == Long.class) return val.asLong();
        if (target == double.class || target == Double.class) return val.asDouble();
        if (target == boolean.class || target == Boolean.class) return val.asBoolean();
        if (target.isEnum()) {
            return convertToEnum(val, (Class<? extends Enum>) target);
        }

        if (target.isArray()) {
            Class<?> componentType = target.getComponentType();
            List<Object> list = val.asList(v -> v.isNull() ? null : convertValue(v, componentType));
            Object array = java.lang.reflect.Array.newInstance(componentType, list.size());
            for (int i = 0; i < list.size(); i++) {
                java.lang.reflect.Array.set(array, i, list.get(i));
            }
            return array;
        }

        if (List.class.isAssignableFrom(target)) {
            return val.asList(v -> v.isNull() ? null : v.asObject());
        }

        return val.asObject();
    }

    private static <E extends Enum<E>> E convertToEnum(Value val, Class<E> enumClass) {
        if (val.isNull()) return null;

        try {
            String strValue = val.asString();
            return Enum.valueOf(enumClass, strValue);
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException(
                    "Failed to convert value '" + val + "' to enum " + enumClass.getName(), e
            );
        }
    }

    /**
     * Sets field value, handling accessibility.
     */
    private static void setFieldValue(Object obj, Field f, Object value) {
        try {
            f.setAccessible(true);
            f.set(obj, value);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to set field " + f.getName(), e);
        }
    }

    /**
     * Gets field value, handling accessibility.
     */
    private static Object getFieldValue(Object obj, Field f) {
        try {
            f.setAccessible(true);
            return f.get(obj);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to get field " + f.getName(), e);
        }
    }

    /**
     * Returns the Neo4j relationship type for this instance.
     */
    public String getType() {
        Relation ann = this.getClass().getAnnotation(Relation.class);
        return ann != null ? ann.type() : this.getClass().getSimpleName();
    }

    /**
     * Returns the direction of the relationship.
     */
    public Relation.Direction getDirection() {
        Relation ann = this.getClass().getAnnotation(Relation.class);
        return ann != null ? ann.direction() : Relation.Direction.OUTGOING;
    }

    private static String resolvePropertyName(Field f) {
        Neo4jProperty ann = f.getAnnotation(Neo4jProperty.class);
        return ann != null ? ann.value() : f.getName();
    }

    /**
     * Returns properties as a Map suitable for use as Cypher parameters.
     * This is the SAFE way to pass properties to Neo4j.
     */
    public Map<String, Object> toPropertiesMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        for (Field f : getMappableFields(this.getClass())) {
            Object value = getFieldValue(this, f);
            if (value != null) {
                String key = resolvePropertyName(f);
                if (value instanceof Enum) {
                    map.put(key, value.toString());
                } else {
                    map.put(key, value);
                }
            }
        }
        return map;
    }

    /**
     * Returns Cypher relationship pattern with parameter placeholder.
     * Example for OUTGOING: -[r:KNOWS $rprops]->
     * Example for INCOMING: <-[r:KNOWS $rprops]-
     * Example for UNDIRECTED: -[r:KNOWS $rprops]-
     * <p>
     * Use with: Map.of("rprops", relation.toPropertiesMap())
     */
    public String toNeo4jPattern(String alias) {
        String aliasStr = alias == null || alias.isEmpty() ? "" : alias;
        String propsParam = "$" + aliasStr + "props";

        return switch (getDirection()) {
            case OUTGOING -> String.format("-[%s:%s %s]->", aliasStr, getType(), propsParam);
            case INCOMING -> String.format("<-[%s:%s %s]-", aliasStr, getType(), propsParam);
            case UNDIRECTED -> String.format("-[%s:%s %s]-", aliasStr, getType(), propsParam);
        };
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
     * DEPRECATED: Use toNeo4jPattern() with parameterized queries instead.
     *
     * @deprecated Use {@link #toNeo4jPattern(String)} with parameterized queries
     */
    @Deprecated
    public String toNeo4j(String alias) {
        return buildNeo4jRelation(toPropertiesMap(), alias);
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
    private String buildNeo4jRelation(Map<String, Object> props, String alias) {
        StringBuilder sb = new StringBuilder();
        String propsStr = props.isEmpty() ? "" : " " + toCypherMap(props);
        switch (getDirection()) {
            case OUTGOING -> sb.append("-[").append(alias).append(":").append(getType()).append(propsStr).append("]->");
            case INCOMING -> sb.append("<-[").append(alias).append(":").append(getType()).append(propsStr).append("]-");
            case UNDIRECTED ->
                    sb.append("-[").append(alias).append(":").append(getType()).append(propsStr).append("]-");
        }
        return sb.toString();
    }

    /**
     * DEPRECATED: Creates inline Cypher map.
     * For backwards compatibility only - use parameterized queries instead.
     */
    @Deprecated
    private String toCypherMap(Map<String, Object> props) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Object> e : props.entrySet()) {
            if (!first) sb.append(", ");
            first = false;
            // Escape property name
            String escapedKey = escapeIdentifier(e.getKey());
            sb.append(escapedKey).append(": ").append(toLiteral(e.getValue()));
        }
        sb.append("}");
        return sb.toString();
    }

    /**
     * Escapes a Cypher identifier (property name).
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
    private String toLiteral(Object v) {
        if (v == null) return "null";
        if (v instanceof Enum<?> enumValue) {
            String escaped = enumValue.name().replace("\\", "\\\\").replace("'", "\\'");
            return "'" + escaped + "'";
        }
        if (v instanceof Number || v instanceof Boolean) return v.toString();
        if (v instanceof Collection<?> col) {
            StringBuilder sb = new StringBuilder("[");
            boolean first = true;
            for (Object x : col) {
                if (!first) sb.append(", ");
                first = false;
                sb.append(toLiteral(x));
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
                sb.append(toLiteral(x));
            }
            sb.append("]");
            return sb.toString();
        }
        String escaped = v.toString().replace("\\", "\\\\").replace("'", "\\'");
        return "'" + escaped + "'";
    }
}