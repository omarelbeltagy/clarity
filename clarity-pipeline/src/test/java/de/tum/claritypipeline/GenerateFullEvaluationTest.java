package de.tum.claritypipeline;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.ClassificationProperties;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.model.core.Taxonomy;
import de.tum.clarityutils.ModelEvaluator;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.xssf.usermodel.XSSFCellStyle;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Objects;

public class GenerateFullEvaluationTest {
    
    private static final String EXCEL_PATH = "src/test/resources/classification_evaluation.xlsx";
    private final Logger log = org.slf4j.LoggerFactory.getLogger(GenerateFullEvaluationTest.class);
    /**
     * Service performing evaluation export operations.
     */
    private final Neo4jClient client = new Neo4jClient();
    
    /**
     * Default constructor.
     *
     * @throws IOException if exporter initialization fails
     */
    public GenerateFullEvaluationTest() throws IOException {}
    
    private List<ClassificationProperties> getAllClassificationProperties() throws IOException {
        String query = """
                MATCH (n:ClassificationProperties)
                RETURN n;
                """;
        return client.executeQuery(query, ClassificationProperties.class);
    }
    
    private void createCell(XSSFRow row, int column, String value, XSSFCellStyle style) {
        Cell cell = row.createCell(column);
        cell.setCellValue(value != null ? value : "");
        cell.setCellStyle(style);
    }
    
    private void createNumericCell(XSSFRow row, int column, double value, XSSFCellStyle style) {
        Cell cell = row.createCell(column);
        cell.setCellValue(value);
        cell.setCellStyle(style);
    }
    
    private void autoSizeColumns(XSSFSheet sheet) {
        for (int i = 0; i < 10; i++) {
            sheet.autoSizeColumn(i);
        }
    }
    
    private void writeWorkbook(XSSFWorkbook workbook, String path) throws IOException {
        try (FileOutputStream fileOut = new FileOutputStream(path)) {
            workbook.write(fileOut);
        }
    }
    
    private void loadClassificationPropertiesMetadata(ClassificationProperties classificationProperties)
            throws IOException {
        String query = String.format("""
                                             MATCH (cp:ClassificationProperties)--(n:Classification)
                                             WHERE elementId(cp) = '%s'
                                             RETURN n;
                                             """,
                                     classificationProperties.getElementId());
        ClassificationProperties.Classification classification = client.executeQuery(query,
                                                                                     ClassificationProperties.Classification.class)
                                                                       .stream()
                                                                       .findFirst()
                                                                       .orElse(null);
        if (classification == null) {
            log.warn("No classification metadata found for ClassificationProperties with elementId {}",
                     classificationProperties.getElementId());
            throw new RuntimeException("Classification metadata not found for ClassificationProperties with elementId "
                                               + classificationProperties.getElementId());
        } else {
            classificationProperties.setClassification(classification);
        }
        String taxonomyQuery = String.format("""
                                                     MATCH (cp:ClassificationProperties)--(n:Taxonomy)
                                                     WHERE elementId(cp) = '%s'
                                                     RETURN n;
                                                     """, classificationProperties.getElementId());
        Taxonomy taxonomy = client.executeQuery(taxonomyQuery, Taxonomy.class)
                                  .stream()
                                  .findFirst()
                                  .orElse(null);
        if (taxonomy == null) {
            log.warn("No taxonomy found for ClassificationProperties with elementId {}",
                     classificationProperties.getElementId());
            throw new RuntimeException("Taxonomy not found for ClassificationProperties with elementId "
                                               + classificationProperties.getElementId());
        } else {
            classificationProperties.taxonomy = taxonomy;
        }
        
        List<Taxonomy.Category> categories = client.executeQuery(String.format("""
                                                                                       MATCH (t:Taxonomy)--(n:Category)
                                                                                       WHERE elementId(t) = '%s'
                                                                                       RETURN n;
                                                                                       """, taxonomy.getElementId()),
                                                                 Taxonomy.Category.class);
        if (categories.isEmpty()) {
            log.warn("No categories found for Taxonomy with elementId {}",
                     taxonomy.getElementId());
            throw new RuntimeException("No categories found for Taxonomy with elementId " + taxonomy.getElementId());
        } else {
            taxonomy.categories = categories;
        }
    }
    
    private EvaluationResult generateEvaluation(ClassificationProperties properties) throws IOException {
        if (!(properties.getTaxonomy().getCategories().size() == 3
                || properties.getTaxonomy().getCategories().size() == 9)) {
            log.info(
                    "Only taxonomies with 3 or 9 categories are supported for evaluation generation. Skipping "
                            + "evaluation for classification run {} of {}",
                    properties.getVersion(), properties.getName());
            return null;
        }
        log.info("Generating evaluation for classification run {} of {}", properties.getVersion(),
                 properties.getName());
        String query = String.format("""
                                             MATCH (n:%s)--(cr:%s)--(c:%s)
                                             WHERE cr.version = '%s'
                                             AND c.name = '%s'
                                             RETURN n
                                             """,
                                     Neo4jNode.getLabel(ClassificationResult.class),
                                     Neo4jNode.getLabel(ClassificationProperties.class),
                                     Neo4jNode.getLabel(ClassificationProperties.Classification.class),
                                     properties.getVersion(),
                                     properties.getClassification().getName());
        List<ClassificationResult> results = client.executeQuery(query,
                                                                 ClassificationResult.class);
        log.info("Found {} classification results for evaluation", results.size());
        if (results.isEmpty()) {
            log.warn("No classification results found for classification run {}. Cannot generate evaluation.",
                     properties.getVersion());
            return null;
        }
        
        List<String[]> predictionsAndExpected =
                results.parallelStream()
                       .map(result -> {
                           String findQAQuery = String.format("""
                                                                      MATCH (cr:%s)--(n:%s)
                                                                      WHERE elementId(cr) = '%s'
                                                                      RETURN n
                                                                      """,
                                                              Neo4jNode.getLabel(ClassificationResult.class),
                                                              Neo4jNode.getLabel(QA.class),
                                                              result.getElementId());
                           
                           QA qa = client.executeQuery(findQAQuery, QA.class)
                                         .stream()
                                         .findFirst()
                                         .orElse(null);
                           
                           if (qa == null) {
                               log.warn("No QA found for classification result {}. Could not generate evaluation",
                                        result.getElementId());
                               return null;
                           }
                           Taxonomy.Category predicted = properties.getTaxonomy().getCategories().stream()
                                                                   .filter(c ->
                                                                                   c.getName()
                                                                                    .equals(result.getName()))
                                                                   .findFirst()
                                                                   .orElse(null);
                           if (predicted == null) {
                               log.warn(
                                       "No category found for classification result {} with name {}. Could not "
                                               + "generate evaluation",
                                       result.getElementId(), result.getName());
                               throw new RuntimeException(
                                       "No category found for classification result " + result.getElementId()
                                               + " with name " + result.getName());
                           }
                           String predictedLabelEvasion = null;
                           String predictedLabelClarity;
                           if (properties.getTaxonomy().getCategories().size() == 3) {
                               predictedLabelClarity = predicted.getName();
                           } else {
                               predictedLabelEvasion = predicted.getName();
                               predictedLabelClarity = predicted.getMapTo();
                           }
                           
                           if (predictedLabelClarity == null) {
                               log.error(
                                       "Predicted label for clarity is null for classification result {}. Cannot "
                                               + "generate evaluation.",
                                       result.getElementId());
                               throw new RuntimeException(
                                       "Predicted label for clarity is null for classification result "
                                               + result.getElementId());
                           }
                           String expectedLabelEvasion = null;
                           if (predictedLabelEvasion != null) {
                               if (predictedLabelEvasion.equals(qa.getAnnotator1())) {
                                   expectedLabelEvasion = qa.getAnnotator1();
                               } else if (predictedLabelEvasion.equals(qa.getAnnotator2())) {
                                   expectedLabelEvasion = qa.getAnnotator2();
                               } else if (predictedLabelEvasion.equals(qa.getAnnotator3())) {
                                   expectedLabelEvasion = qa.getAnnotator3();
                               } else {
                                   expectedLabelEvasion = qa.getAnnotator1();
                               }
                           }
                           
                           String expectedLabelClarity = qa.getClarityLabel();
                           if (expectedLabelClarity == null) {
                               log.error(
                                       "Expected label for clarity is null for QA {} linked to classification result "
                                               + "{}. Cannot "
                                               + "generate evaluation.",
                                       qa.getElementId(), result.getElementId());
                               throw new RuntimeException(
                                       "Expected label for clarity is null for QA " + qa.getElementId()
                                               + " linked to classification result " + result.getElementId());
                           }
                           return new String[]{predictedLabelClarity, expectedLabelClarity, predictedLabelEvasion,
                                               expectedLabelEvasion,
                                               qa.getAnnotator1(),
                                               qa.getAnnotator2(),
                                               qa.getAnnotator3()
                           };
                       })
                       .filter(Objects::nonNull)
                       .toList();
        
        List<String> predictionsClarity = predictionsAndExpected.stream()
                                                                .map(arr -> arr[0])
                                                                .toList();
        
        List<String> expectedClarity = predictionsAndExpected.stream()
                                                             .map(arr -> arr[1])
                                                             .toList();
        
        List<String> predictionsEvasion = predictionsAndExpected.stream()
                                                                .filter(arr -> arr[2] != null)
                                                                .map(arr -> arr[2])
                                                                .toList();
        
        List<String> expectedEvasion = predictionsAndExpected.stream()
                                                             .filter(arr -> arr[3] != null)
                                                             .map(arr -> arr[3])
                                                             .toList();
        
        List<List<String>> annotationEvasion = predictionsAndExpected.stream()
                                                                     .filter(arr -> arr[2] != null)
                                                                     .map(arr -> List.of(arr[4], arr[5], arr[6]))
                                                                     .toList();
        
        List<String> labelsClarity = List.of("Clear Reply", "Ambivalent", "Clear Non-Reply");
        List<String> labelsEvasion = List.of(
                "Explicit",
                "Implicit",
                "General",
                "Partial/half-answer",
                "Dodging",
                "Deflection",
                "Declining to answer",
                "Claims ignorance",
                "Clarification"
        );
        
        try {
            ModelEvaluator evaluatorClarity = new ModelEvaluator(labelsClarity, predictionsClarity, expectedClarity);
            Double accuracy = evaluatorClarity.accuracy();
            Double precision = evaluatorClarity.precision();
            Double recall = evaluatorClarity.recall();
            Double microF1 = evaluatorClarity.microF1();
            Double macroF1 = evaluatorClarity.macroF1();
            
            Double accuracyEvasion = null;
            Double macroF1Evasion = null;
            if (!predictionsEvasion.isEmpty()
                    && !expectedEvasion.isEmpty() && predictionsEvasion.size() == expectedEvasion.size()) {
                ModelEvaluator evaluatorEvasion = new ModelEvaluator(labelsEvasion, predictionsEvasion,
                                                                     expectedEvasion);
                accuracyEvasion = evaluatorEvasion.accuracy();
                macroF1Evasion = evaluatorEvasion.multiLabelMacroF1(annotationEvasion);
            }
            
            return new EvaluationResult(accuracy, precision, recall, microF1, macroF1, accuracyEvasion, macroF1Evasion);
        } catch (Exception e) {
            log.error("Error while evaluating classification run {}", properties.getVersion(), e);
            throw new RuntimeException("Error while evaluating classification run " + properties.getVersion(), e);
        }
    }
    
    private void exportEvaluationResultsToExcel(
            List<ClassificationProperties> propertiesList,
            List<EvaluationResult> results,
            String filePath
    ) throws IOException {
        XSSFWorkbook workbook = new XSSFWorkbook();
        XSSFSheet sheet = workbook.createSheet("Evaluation Results");
        
        // Create header row
        XSSFRow headerRow = sheet.createRow(0);
        String[] headers = {"Version", "Name", "Accuracy Clarity", "Precision Clarity", "Recall Clarity",
                            "Micro F1 Clarity", "Macro F1 Clarity", "Accuracy Evasion", "Macro F1 Evasion"};
        for (int i = 0; i < headers.length; i++) {
            createCell(headerRow, i, headers[i], null);
        }
        
        // Fill data rows
        int rowIndex = 1;
        for (int i = 0; i < propertiesList.size(); i++) {
            ClassificationProperties properties = propertiesList.get(i);
            EvaluationResult result = results.get(i);
            if (result == null) {
                continue;
            }
            XSSFRow row = sheet.createRow(rowIndex++);
            createCell(row, 0, properties.getVersion(), null);
            createCell(row, 1, properties.getName(), null);
            createNumericCell(row, 2, result.accuracyClarity(), null);
            createNumericCell(row, 3, result.precisionClarity(), null);
            createNumericCell(row, 4, result.recallClarity(), null);
            createNumericCell(row, 5, result.microF1Clarity(), null);
            createNumericCell(row, 6, result.macroF1Clarity(), null);
            if (result.accuracyEvasion() != null) {
                createNumericCell(row, 7, result.accuracyEvasion(), null);
            }
            if (result.macroF1Evasion() != null) {
                createNumericCell(row, 8, result.macroF1Evasion(), null);
            }
        }
        
        autoSizeColumns(sheet);
        writeWorkbook(workbook, filePath);
    }
    
    @Test
    public void testExportAllToExcel() throws IOException {
        List<ClassificationProperties> allProperties = getAllClassificationProperties();
        log.info("Found {} classification runs to evaluate", allProperties.size());
        List<EvaluationResult> allResults = allProperties.stream()
                                                         .map(properties -> {
                                                             try {
                                                                 loadClassificationPropertiesMetadata(properties);
                                                                 return generateEvaluation(properties);
                                                             } catch (Exception e) {
                                                                 log.error("Error processing classification run {}: {}",
                                                                           properties.getVersion(), e.getMessage(), e);
                                                                 return null;
                                                             }
                                                         })
                                                         .toList();
        exportEvaluationResultsToExcel(allProperties, allResults, EXCEL_PATH);
    }
    
    record EvaluationResult(Double accuracyClarity,
                            Double precisionClarity,
                            Double recallClarity,
                            Double microF1Clarity,
                            Double macroF1Clarity,
                            Double accuracyEvasion,
                            Double macroF1Evasion
    
    ) {
    }
}
