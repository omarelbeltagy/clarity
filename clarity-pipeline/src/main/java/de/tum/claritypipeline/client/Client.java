package de.tum.claritypipeline.client;

import de.tum.claritypipeline.model.properties.ClassificationProperties;
import de.tum.claritypipeline.model.properties.ModelConfig;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Arrays;
import java.util.Locale;
import java.util.stream.Collectors;

public interface Client {
    String makeRequest(String prompt);

    <T> T makeRequest(String prompt, Class<T> responseType);

    static Client create(ModelConfig modelConfig) {
        ModelProvider provider = parseProvider(modelConfig.getProvider());
        return provider.createClient(modelConfig);
    }

    /**
     * Parse and validate the provider name into a {@link ModelProvider} enum.
     *
     * @param providerName the raw provider prefix (e.g., "openai")
     * @return the matching {@link ModelProvider}
     * @throws IllegalArgumentException if provider is not supported
     */
    private static ModelProvider parseProvider(String providerName) {
        ModelProvider provider = ModelProvider.fromValue(providerName);
        if (provider == null) {
            throw new IllegalArgumentException(String.format(
                    "Invalid model provider '%s'. Allowed providers are: %s",
                    providerName,
                    ModelProvider.getAllowedProviders()));
        }
        return provider;
    }

    /**
     * Enum of supported model providers and a factory method to create provider-specific clients.
     *
     * <p>Each enum constant contains a human-readable name and a factory function that
     * accepts {@link ClassificationProperties} and returns a {@link Client}.
     */
    @Getter
    @AllArgsConstructor
    enum ModelProvider {
        ANTHROPIC("anthropic", AnthropicClient::new),
        LOCAL("local", LocalClient::new),
        TOGETHER("together", TogetherClient::new),
        OPENAI("openai", OpenAIClient::new);

        /**
         * Canonical provider name as used in configuration (lowercase).
         */
        private final String name;

        /**
         * Factory used to instantiate provider-specific {@link Client} objects.
         */
        private final ClientFactory factory;

        /**
         * Attempt to map a raw provider string to a {@link ModelProvider} enum.
         *
         * <p>Normalization: uppercase, remove whitespace, hyphens and underscores.
         *
         * @param raw raw provider string (may be null)
         * @return matching de.tum.claritypipeline.model.ModelProvider or null if none matches
         */
        public static ModelProvider fromValue(String raw) {
            if (raw == null) return null;
            String normalized = raw.toUpperCase(Locale.ROOT)
                                   .replaceAll("[\\s\\-_]", "");
            try {
                return ModelProvider.valueOf(normalized);
            } catch (IllegalArgumentException e) {
                return null;
            }
        }

        /**
         * Return a comma-separated list of allowed provider names for error messages.
         *
         * @return CSV list of provider names
         */
        public static String getAllowedProviders() {
            return Arrays.stream(ModelProvider.values())
                         .map(ModelProvider::getName)
                         .collect(Collectors.joining(", "));
        }

        /**
         * Create a client instance for this provider using the supplied properties.
         *
         * @param config the model properties
         * @return a new Client instance for the provider
         */
        public Client createClient(ModelConfig config) {
            return factory.create(config);
        }

        /**
         * Functional interface defining the factory signature for clients.
         */
        @FunctionalInterface
        private interface ClientFactory {
            /**
             * Create a client using the provided properties.
             *
             * @param config the model properties
             * @return a concrete Classifier instance
             */
            Client create(ModelConfig config);
        }
    }
}
