package de.tum.clarityutils;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.victools.jsonschema.generator.OptionPreset;
import com.github.victools.jsonschema.generator.SchemaGenerator;
import com.github.victools.jsonschema.generator.SchemaGeneratorConfigBuilder;
import com.github.victools.jsonschema.generator.SchemaVersion;
import com.github.victools.jsonschema.module.jackson.JacksonModule;
import lombok.*;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;

public class JsonSchemeTest {
    private final Logger log = org.slf4j.LoggerFactory.getLogger(JsonSchemeTest.class);

    @Test
    public void testConfigurationPropertiesScheme() throws JsonProcessingException {
        SchemaGeneratorConfigBuilder configBuilder = new SchemaGeneratorConfigBuilder(
                SchemaVersion.DRAFT_2020_12, OptionPreset.PLAIN_JSON)
                .with(new JacksonModule());
        SchemaGenerator generator = new SchemaGenerator(configBuilder.build());

        ObjectMapper mapper = new ObjectMapper();
        JsonNode jsonSchema = generator.generateSchema(ConfigurationProperties.class);
        log.info(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(jsonSchema));
    }

    private enum ResponseFormat {
        @JsonProperty("json_object")
        JSON_OBJECT,

        @JsonProperty("text")
        TEXT
    }

    @Getter
    @Setter
    @AllArgsConstructor
    @NoArgsConstructor
    @Builder
    private static class ConfigurationProperties {
        @JsonProperty("name")
        @JsonPropertyDescription("The name of the configuration")
        public String name;
        @JsonProperty("version")
        public String version;
        @JsonProperty("model")
        public String prompt;
        @JsonProperty(value = "attempts", defaultValue = "5")
        @JsonPropertyDescription("The model to be used for classification")
        private int attempts;
        @JsonProperty(value = "max-tokens", defaultValue = "4096")
        private int maxTokens;
        @JsonProperty(value = "top-p", defaultValue = "0.5")
        private double topP;
        @JsonProperty(value = "temperature", defaultValue = "0.7")
        private double temperature;
        @JsonProperty(value = "response-format", defaultValue = "json_object")
        private ResponseFormat responseFormat;
    }
}
