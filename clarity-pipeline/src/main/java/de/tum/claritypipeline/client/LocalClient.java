package de.tum.claritypipeline.client;

import de.tum.claritypipeline.model.properties.ModelConfig;
import de.tum.clarityutils.SerializationUtils;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import okhttp3.*;
import org.slf4j.Logger;

import java.nio.charset.StandardCharsets;

@AllArgsConstructor
@Getter
@Setter
public class LocalClient implements Client {
    private final Logger log = org.slf4j.LoggerFactory.getLogger(LocalClient.class);

    private final ModelConfig properties;

    @Override
    public String makeRequest(String body) {
        return makeRequest(body, String.class);
    }

    @Override
    public <T> T makeRequest(String body, Class<T> clazz) {
        try {
            OkHttpClient client = new OkHttpClient();

            MediaType JSON = MediaType.parse("application/json; charset=utf-8");

            RequestBody requestBody = RequestBody.create(
                    body.getBytes(StandardCharsets.UTF_8),
                    JSON
            );

            Request request = new Request.Builder()
                    .url(properties.getName())
                    .post(requestBody)
                    .build();

            try (Response response = client.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new RuntimeException("Unexpected code " + response);
                }
                try (ResponseBody responseBody = response.body()) {
                    assert responseBody != null;
                    return SerializationUtils.deserialize(
                            responseBody.string(), clazz);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Error during classification", e);
        }
    }
}