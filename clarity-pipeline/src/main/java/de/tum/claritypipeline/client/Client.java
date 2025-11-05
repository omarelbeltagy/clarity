package de.tum.claritypipeline.client;

import de.tum.claritypipeline.model.config.ClassificationProperties;
import de.tum.claritypipeline.model.config.ModelProperties;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Arrays;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * Client defines the contract for model clients used to send prompts and receive responses.
 * Implementations must provide makeRequest methods for text and structured output.
 */
public interface Client {
    /**
     * Create a concrete Client instance based on the provided model configuration.
     *
     * @param modelConfig model configuration containing provider and model settings
     * @return concrete Client implementation
     * @throws IllegalArgumentException if provider is invalid
     */
    static Client create(ModelProperties modelConfig) {
        ModelProvider provider = parseProvider(modelConfig.getProvider());
        return provider.createClient(modelConfig);
    }

    /**
     * Send a prompt and receive a plain text response.
     *
     * @param prompt prompt text
     * @return response as String or null on error
     */
    String makeRequest(String prompt);

    /**
     * Send a prompt and parse the response into the specified type.
     *
     * @param prompt       prompt text
     * @param responseType expected response class
     * @param <T>          response type
     * @return parsed response instance or null on error
     */
    <T> T makeRequest(String prompt, Class<T> responseType);

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
         * @return matching ModelProvider or null if none matches
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
        public Client createClient(ModelProperties config) {
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
             * @return a concrete Client instance
             */
            Client create(ModelProperties config);
        }
    }
}
