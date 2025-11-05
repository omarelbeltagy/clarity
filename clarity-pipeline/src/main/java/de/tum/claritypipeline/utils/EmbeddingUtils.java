package de.tum.claritypipeline.utils;

import com.openai.models.embeddings.CreateEmbeddingResponse;
import com.openai.models.embeddings.EmbeddingCreateParams;
import com.openai.models.embeddings.EmbeddingModel;
import org.slf4j.Logger;

import java.util.List;

public class EmbeddingUtils {
    private final Logger log = org.slf4j.LoggerFactory.getLogger(EmbeddingUtils.class);

    /**
     * Converts a list of Float to a double array.
     *
     * @param list List of Float values
     * @return Array of double values
     */
    private static double[] toDoubleArray(List<Float> list) {
        double[] arr = new double[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }

    /**
     * Creates an embedding for the given text using the OpenAI API.
     *
     * @param text Text to create an embedding for
     * @return Embedding as a double array
     */
    public static double[] createEmbedding(String text, EmbeddingModel model, com.openai.client.OpenAIClient client) {
        EmbeddingCreateParams params = EmbeddingCreateParams.builder()
                                                            .model(model)
                                                            .input(text)
                                                            .build();
        CreateEmbeddingResponse response = client.embeddings().create(params);
        return toDoubleArray(response.data().getFirst().embedding());
    }
}
