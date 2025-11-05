package de.tum.claritypipeline;

import de.tum.claritypipeline.model.config.DatasetType;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.service.ClassificationPipeline;
import de.tum.claritypipeline.service.DatasetGraphImporter;
import de.tum.claritypipeline.service.DatasetReader;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Tests the overall classification pipeline.
 *
 * <p>Includes dataset ingestion, graph import, and running classification experiments using
 * different model/agent configurations defined by properties files.</p>
 */
public class PipelineTest {
    /**
     * Service for reading dataset JSON files from test resources.
     */
    private final DatasetReader datasetReaderService = new DatasetReader();

    /**
     * Service for importing QA records into an internal graph representation.
     */
    private final DatasetGraphImporter datasetGraphImporter = new DatasetGraphImporter();

    /**
     * Default constructor.
     *
     * @throws IOException if initialization fails
     */
    public PipelineTest() throws IOException {}

    /**
     * Import training, validation and test datasets into the graph importer.
     *
     * <p>This merges multiple dataset splits into a single in-memory collection and
     * imports them into the neo4j database.</p>
     */
    @Test
    public void testImportDatasets() {
        List<QA> data = new ArrayList<>();
        data.addAll(datasetReaderService.readDataset(DatasetType.TRAIN));
        data.addAll(datasetReaderService.readDataset(DatasetType.VALID));
        data.addAll(datasetReaderService.readDataset(DatasetType.TEST));
        datasetGraphImporter.importDataset(data);
    }

    /**
     * Run classification using a single properties file.
     *
     * @throws IOException if the pipeline fails to initialize or run
     */
    @Test
    public void testClassify() throws IOException {
        ClassificationPipeline classificationPipeline = new ClassificationPipeline(
                "src/test/resources/properties/few-shot/Llama-3.3-70B.yaml");
        classificationPipeline.classify();
    }

    /**
     * Run classification experiments for different encoder-based configurations in parallel.
     *
     * @throws IOException if any pipeline initialization fails
     */
    @Test
    public void testClassifyEncoder() throws IOException {
        List<ClassificationPipeline> pipelines = new ArrayList<>();
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/encoder/roberta-base.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/encoder/roberta-large.yaml"));

        pipelines.parallelStream().forEach(ClassificationPipeline::classify);
    }


    @Test
    public void testClassifyFineTuned() throws IOException {
        List<ClassificationPipeline> pipelines = new ArrayList<>();
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/fine-tune/Llama-3-8B-LoRA.yaml"));

        pipelines.parallelStream().forEach(ClassificationPipeline::classify);
    }

    /**
     * Run classification experiments for different Llama-based configurations in parallel.
     *
     * @throws IOException if any pipeline initialization fails
     */
    @Test
    public void testClassifyLlama() throws IOException {
        List<ClassificationPipeline> pipelines = new ArrayList<>();
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/Llama-3.3-70B.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/Llama-4-Maverick-17B-128E.yaml"));

        pipelines.parallelStream().forEach(ClassificationPipeline::classify);
    }

    /**
     * Run classification experiments for Claude model variants.
     *
     * @throws IOException if any pipeline initialization fails
     */
    @Test
    public void testClassifyClaude() throws IOException {
        List<ClassificationPipeline> pipelines = new ArrayList<>();
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/claude-haiku-4.5.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/claude-sonnet-4.5.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/claude-opus-4.1.yaml"));

        pipelines.parallelStream().forEach(ClassificationPipeline::classify);
    }

    /**
     * Run classification experiments for DeepSeek model variants.
     *
     * @throws IOException if any pipeline initialization fails
     */
    @Test
    public void testClassifyDeepSeek() throws IOException {
        List<ClassificationPipeline> pipelines = new ArrayList<>();
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/DeepSeek-R1-0528-tput.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/DeepSeek-R1-Distill-Qwen-14B.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/DeepSeek-R1-Distill-Llama-70B.yaml"));

        pipelines.parallelStream().forEach(ClassificationPipeline::classify);
    }

    /**
     * Run a variety of OpenAI / OSS GPT model configurations for classification.
     *
     * @throws IOException if any pipeline initialization fails
     */
    @Test
    public void testClassifyOpenAi() throws IOException {
        List<ClassificationPipeline> pipelines = new ArrayList<>();
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/gpt-4.1.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/gpt-4.1-mini.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/gpt-oss-20b.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/gpt-oss-120b.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/gpt-5.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/gpt-5-mini.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/gpt-5-nano.yaml"));
        pipelines.add(new ClassificationPipeline(
                "src/test/resources/properties/few-shot/o3.yaml"));

        pipelines.parallelStream().forEach(ClassificationPipeline::classify);
    }
}
