package de.tum.claritypipeline.client;

public interface Client {
    String makeRequest(String prompt);

    <T> T makeRequest(String prompt, Class<T> responseType);
}
