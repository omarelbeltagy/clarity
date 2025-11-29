package de.tum.claritypipeline.service;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.credential.BearerTokenCredential;
import com.openai.models.embeddings.EmbeddingModel;
import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.clarityneo4j.model.Neo4jEmbeddingSearchResult;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.utils.EmbeddingUtils;
import de.tum.clarityutils.EnvLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class EmbeddingService {

    private static final Logger log = LoggerFactory.getLogger(EmbeddingService.class);
    private static EmbeddingService instance;
    private final OpenAIClient client;
    private final EmbeddingModel model;
    private final Neo4jClient neo4jClient;

    private EmbeddingService(Neo4jCredentials neo4jCredentials, String modelName) {
        String apiKey = EnvLoader.get("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isEmpty()) {
            throw new IllegalStateException(
                    "OPENAI_API_KEY environment variable is not set. Please set it to use OpenAIClassifier.");
        }
        this.client = new OpenAIOkHttpClient.Builder().credential(
                BearerTokenCredential.create(apiKey)).build();
        this.neo4jClient = new Neo4jClient(neo4jCredentials);
        this.model = EmbeddingModel.of(modelName);
    }

    public static synchronized void initialize(Neo4jCredentials neo4jCredentials, String modelName) {
        if (instance != null) {
            log.warn("EmbeddingService was already initialized. Skipping Initialization.");
            return;
        }
        instance = new EmbeddingService(neo4jCredentials, modelName);
        EmbeddingUtils.ensureEmbeddingIndicesExist(instance.neo4jClient);
    }

    public static EmbeddingService getInstance() {
        if (instance == null) {
            throw new IllegalStateException("EmbeddingService was not yet initialized");
        }
        return instance;
    }

    // ----------------------------------------
    // Restlicher Code bleibt gleich
    // ----------------------------------------

    public double[] generateEmbeddings(String text) {
        return EmbeddingUtils.createEmbedding(text, model, client);
    }

    public void generateEmbeddingsForQAs(String query) {
        List<QA> qas = neo4jClient.executeQuery(query, QA.class);
        AtomicInteger counter = new AtomicInteger(0);
        qas.parallelStream().forEach(qa -> {
            log.info("Generating embeddings {}/{}", counter.addAndGet(1), qas.size());
            if (qa.getQuestionEmbedding() != null && qa.getAnswerEmbedding() != null
                    && qa.getQuestionAnswerEmbedding() != null) {
                log.info("Embeddings already exist for QA. Skipping...");
                return;
            }
            if (qa.getQuestionEmbedding() == null) {
                double[] questionEmbedding = EmbeddingUtils.createEmbedding(qa.getQuestion(), model, client);
                qa.setQuestionEmbedding(questionEmbedding);
            }
            if (qa.getAnswerEmbedding() == null) {
                double[] answerEmbedding = EmbeddingUtils.createEmbedding(qa.getInterviewAnswer(), model, client);
                qa.setAnswerEmbedding(answerEmbedding);
            }
            if (qa.getQuestionAnswerEmbedding() == null) {
                String combinedText = qa.getQuestion() + " " + qa.getInterviewAnswer();
                double[] questionAnswerEmbedding = EmbeddingUtils.createEmbedding(combinedText, model, client);
                qa.setQuestionAnswerEmbedding(questionAnswerEmbedding);
            }
            neo4jClient.updateNode(qa);
        });
    }

    public <T extends Neo4jNode> List<Neo4jEmbeddingSearchResult<T>> searchSimilar(
            String index, double[] embedding, int k, Class<T> nodeClass
    ) {
        return neo4jClient.similaritySearch(index, embedding, k, nodeClass);
    }
}