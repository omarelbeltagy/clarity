package de.tum.claritypipeline.service;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.core.Neo4jNode;
import de.tum.clarityneo4j.core.Neo4jRelation;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.claritypipeline.model.classification.ClassificationResult;
import de.tum.claritypipeline.model.config.ClassificationProperties;
import de.tum.claritypipeline.model.config.EvaluationExportProperties;
import de.tum.claritypipeline.model.core.QA;
import de.tum.claritypipeline.model.core.Taxonomy;
import de.tum.claritypipeline.model.relation.GeneratedBy;
import de.tum.claritypipeline.model.relation.HasClassification;
import de.tum.clarityutils.ModelEvaluator;
import lombok.Builder;
import lombok.Getter;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Utility responsible for exporting evaluation results from Neo4j into an Excel file.
 *
 * <p>This class queries the Neo4j database for Classification / Version nodes,
 * retrieves Evaluation objects from version nodes and writes a tabular XLSX file with
 * configurable formatting options.</p>
 *
 * <p>Creation is performed via factory methods which allow providing Neo4j credentials
 * or custom EvaluationExportOptions.</p>
 */
public class EvaluationExporter {
    private static final Logger log = LoggerFactory.getLogger(EvaluationExporter.class);

    private static final String[] HEADERS = {"Name", "Version", "Accuracy",
                                             "Precision", "Recall", "Macro F1", "Micro F1"
    };

    private final Neo4jClient client;
    private final EvaluationExportProperties options;

    private EvaluationExporter(Neo4jClient client, EvaluationExportProperties options) {
        this.client = client;
        this.options = options;
    }

    /**
     * Create a new exporter using default Neo4j connection settings and default export options.
     *
     * @return a new EvaluationExporter instance
     * @throws IOException if the default Neo4j client could not be initialized
     */
    public static EvaluationExporter create() throws IOException {
        return new EvaluationExporter(new Neo4jClient(), new EvaluationExportProperties());
    }

    /**
     * Create a new exporter with explicit Neo4j credentials and default export options.
     *
     * @param credentials Neo4j credentials to use for the client
     * @return configured EvaluationExporter
     */
    public static EvaluationExporter create(Neo4jCredentials credentials) {
        return new EvaluationExporter(new Neo4jClient(credentials), new EvaluationExportProperties());
    }

    /**
     * Create a new exporter using default Neo4j connection settings and the given options.
     *
     * @param options export formatting and behavior options (may be null)
     * @return configured EvaluationExporter
     * @throws IOException if the default Neo4j client could not be initialized
     */
    public static EvaluationExporter create(EvaluationExportProperties options) throws IOException {
        return new EvaluationExporter(new Neo4jClient(), options);
    }

    /**
     * Create a new exporter with explicit Neo4j credentials and given export options.
     *
     * @param credentials Neo4j credentials for the client
     * @param options     export formatting and behavior options
     * @return configured EvaluationExporter
     */
    public static EvaluationExporter create(Neo4jCredentials credentials, EvaluationExportProperties options) {
        return new EvaluationExporter(new Neo4jClient(credentials), options);
    }

    /**
     * Load Neo4j credentials from a file and create a new exporter.
     *
     * @param credentialsFile path to a credentials file
     * @return configured EvaluationExporter
     * @throws IOException if the credentials file cannot be read
     */
    public static EvaluationExporter fromCredentialsFile(String credentialsFile) throws IOException {
        return create(Neo4jCredentials.load(credentialsFile));
    }

    /**
     * Export evaluation data from the connected Neo4j database to an XLSX file.
     *
     * <p>This method collects classifications and their child nodes, extracts Evaluation objects
     * from version nodes and writes the results to the given file path. On failure a
     * RuntimeException is thrown.</p>
     *
     * @param path output path for the generated Excel file (XLSX)
     */
    public void exportAsExcel(String path) {
        try {
            List<ClassificationProperties.Classification> classifications = client.findNodes(Map.of(),
                                                                                             ClassificationProperties.Classification.class);
            List<ExcelRow> excelRows = readExcelRows(classifications);
            exportToExcelFile(excelRows, path);
        } catch (Exception e) {
            log.error("Failed to export evaluation data to Excel", e);
            throw new RuntimeException("Excel export failed", e);
        }
    }

    public void generateCustomEvaluation(String classificationPropertiesPath) throws IOException {
        ClassificationProperties classificationProperties = ClassificationProperties.load(classificationPropertiesPath);
        generateCustomEvaluation(classificationProperties);
    }

    public void generateCustomEvaluation(ClassificationProperties properties) {
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
                           String predictedLabel;
                           if (properties.getTaxonomy().getMapping() != null && properties.getTaxonomy().getMapping()
                                                                                          .isEnabled()) {
                               Taxonomy.Category category = properties.getTaxonomy().getCategories().stream()
                                                                      .filter(c ->
                                                                                      c.getName()
                                                                                       .equals(result.getName()))
                                                                      .findFirst()
                                                                      .orElse(null);
                               if (category != null) {
                                   predictedLabel = category.getMapTo();
                               } else {
                                   return null;
                               }
                           } else {
                               predictedLabel = result.getName();
                           }
                           String propertyLabel =
                                   (properties.getTaxonomy().getMapping() != null && properties.getTaxonomy()
                                                                                               .getMapping()
                                                                                               .isEnabled())
                                           ? properties.getTaxonomy().getMapping().getLabelProperty()
                                           : properties.getTaxonomy().getLabelProperty();
                           String expectedLabel;
                           try {
                               Field field = qa.getClass().getDeclaredField(propertyLabel);
                               field.setAccessible(true);
                               Object value = field.get(qa);
                               if (value == null) {
                                   return null;
                               }
                               expectedLabel = value.toString();
                           } catch (NoSuchFieldException | IllegalAccessException e) {
                               throw new RuntimeException(e);
                           }
                           if (predictedLabel != null && expectedLabel != null) {
                               return new String[]{predictedLabel, expectedLabel};
                           }
                           return null;
                       })
                       .filter(Objects::nonNull)
                       .toList();

        List<String> predictions = predictionsAndExpected.stream()
                                                         .map(arr -> arr[0])
                                                         .toList();

        List<String> expected = predictionsAndExpected.stream()
                                                      .map(arr -> arr[1])
                                                      .toList();

        List<String> labels;
        if (properties.getTaxonomy().getMapping() != null && properties.getTaxonomy().getMapping().isEnabled()) {
            labels = properties.getTaxonomy().getMapping().getLabels();
        } else {
            labels = properties.getTaxonomy().getCategories()
                               .stream()
                               .map(Taxonomy.Category::getName)
                               .toList();
        }

        try {
            ModelEvaluator evaluator = new ModelEvaluator(labels, predictions, expected);
            log.info("Evaluation Results:");
            double accuracy = evaluator.accuracy();
            log.info("Accuracy: {}", String.format("%.2f", accuracy * 100));
            double precision = evaluator.precision();
            log.info("Precision: {}", String.format("%.2f", precision * 100));
            double recall = evaluator.recall();
            log.info("Recall: {}", String.format("%.2f", recall * 100));
            double microF1 = evaluator.microF1();
            log.info("Micro F1 Score: {}", String.format("%.2f", microF1 * 100));
            double macroF1 = evaluator.macroF1();
            log.info("Macro F1 Score: {}", String.format("%.2f", macroF1 * 100));
        } catch (Exception e) {
            log.error("Error while evaluating classification run {}", properties.getVersion(), e);
        }

    }

    public void exportResult(String classificationPropertiesPath, String outputFile) throws IOException {
        ClassificationProperties classificationProperties = ClassificationProperties.load(classificationPropertiesPath);
        exportResult(classificationProperties, outputFile);
    }

    public void exportResult(ClassificationProperties classificationProperties, String outputFile) throws IOException {
        log.info("Exporting evaluation data for {}({}) to file", classificationProperties.getName(),
                 classificationProperties.getVersion());
        String query = """
                MATCH(qa:%s)-[:%s]->(n:%s)-[:%s]->(m:%s)
                WHERE elementId(m) = $propsNodeId
                RETURN n
                ORDER BY qa.index ASC
                """.formatted(
                Neo4jNode.getLabel(QA.class),
                Neo4jRelation.getType(HasClassification.class),
                Neo4jNode.getLabel(ClassificationResult.class),
                Neo4jRelation.getType(GeneratedBy.class),
                Neo4jNode.getLabel(ClassificationProperties.class)
        );
        List<ClassificationResult> results = client.executeQuery(query, Map.of("propsNodeId",
                                                                               classificationProperties.getElementId()),
                                                                 ClassificationResult.class);

        Path temp = Files.createTempFile("prediction", ".tmp");
        try (BufferedWriter writer = Files.newBufferedWriter(temp)) {
            for (int i = 0; i < results.size(); i++) {
                ClassificationResult result = results.get(i);
                if (classificationProperties.getTaxonomy().getMapping() == null
                        || !classificationProperties.getTaxonomy().getMapping().isEnabled()) {
                    writer.write(result.getName());
                } else {
                    Taxonomy.Category category = classificationProperties.getTaxonomy().getCategories().stream()
                                                                         .filter(c -> c.getName()
                                                                                       .equals(result.getName()))
                                                                         .findFirst().orElse(null);
                    if (category == null) {
                        throw new RuntimeException("Could not find category with name " + result.getName());
                    }
                    String name = category.getMapTo();
                    if (name == null) {
                        throw new RuntimeException("Could not find mapping for category with name " + result.getName());
                    }
                    writer.write(name);
                }
                if (i < results.size() - 1) {
                    writer.newLine();
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        Path zipPath = Paths.get(outputFile);
        try (ZipOutputStream zipOut = new ZipOutputStream(Files.newOutputStream(zipPath))) {

            ZipEntry entry = new ZipEntry("prediction");
            zipOut.putNextEntry(entry);

            Files.copy(temp, zipOut);

            zipOut.closeEntry();
        }
        Files.deleteIfExists(temp);

        log.info("Successfully exported evaluation data to file {}", outputFile);

    }

    /**
     * Read the Evaluation objects from the provided classifications and map them to ExcelRow DTOs.
     *
     * @param classifications list of Classification nodes retrieved from Neo4j
     * @return list of ExcelRow entries to be written to the spreadsheet
     */
    private List<ExcelRow> readExcelRows(List<ClassificationProperties.Classification> classifications) {
        return classifications.stream()
                              .flatMap(classification -> classification.getRuns(client).stream()
                                                                       .map(version -> createExcelRow(
                                                                               classification,
                                                                               version)))
                              .filter(Objects::nonNull)
                              .toList();
    }

    /**
     * Attempt to retrieve an Evaluation instance from a version node via reflection.
     *
     * @param version version node object from which to fetch the Evaluation
     * @return Evaluation instance if available, otherwise null
     */
    private ClassificationProperties.Evaluation getEvaluation(Object version) {
        try {
            return (ClassificationProperties.Evaluation) version.getClass()
                                                                .getMethod("getEvaluation", Neo4jClient.class)
                                                                .invoke(version, client);
        } catch (Exception e) {
            log.warn("Failed to get evaluation for version", e);
            return null;
        }
    }

    private ExcelRow createExcelRow(Object classification, Object version) {
        ClassificationProperties.Evaluation evaluation = getEvaluation(version);
        if (evaluation == null) {
            return null;
        }

        if (shouldRoundValues()) {
            evaluation = roundEvaluation(evaluation);
        }

        return ExcelRow.builder()
                       .name(getClassificationName(classification))
                       .version(getVersionNumber(version))
                       .accuracy(evaluation.getAccuracy())
                       .precision(evaluation.getPrecision())
                       .recall(evaluation.getRecall())
                       .macroF1(evaluation.getMacroF1())
                       .microF1(evaluation.getMicroF1())
                       .build();
    }

    private String getClassificationName(Object classification) {
        try {
            return (String) classification.getClass().getMethod("getName").invoke(classification);
        } catch (Exception e) {
            return "";
        }
    }

    private String getVersionNumber(Object version) {
        try {
            return (String) version.getClass().getMethod("getVersion").invoke(version);
        } catch (Exception e) {
            return "";
        }
    }

    private boolean shouldRoundValues() {
        return options.getRoundToDigits() > 0;
    }

    private ClassificationProperties.Evaluation roundEvaluation(ClassificationProperties.Evaluation eval) {
        double factor = Math.pow(10, options.getRoundToDigits());
        return ClassificationProperties.Evaluation.builder()
                                                  .accuracy(round(eval.getAccuracy(), factor))
                                                  .precision(round(eval.getPrecision(), factor))
                                                  .recall(round(eval.getRecall(), factor))
                                                  .macroF1(round(eval.getMacroF1(), factor))
                                                  .microF1(round(eval.getMicroF1(), factor))
                                                  .build();
    }

    private double round(double value, double factor) {
        return Math.round(value * factor) / factor;
    }

    private void exportToExcelFile(List<ExcelRow> excelRows, String path) {
        try (XSSFWorkbook workbook = new XSSFWorkbook()) {
            XSSFSheet sheet = workbook.createSheet(options.getSheetName());

            StyleHelper styles = new StyleHelper(workbook, options);
            createHeaderRow(sheet, styles);
            populateDataRows(sheet, excelRows, styles);
            autoSizeColumns(sheet);

            writeWorkbook(workbook, path);
            log.info("Excel export successful: {}", path);
        } catch (IOException e) {
            log.error("Failed to export Excel file", e);
            throw new RuntimeException("Failed to write Excel file", e);
        }
    }

    private void createHeaderRow(XSSFSheet sheet, StyleHelper styles) {
        XSSFRow headerRow = sheet.createRow(0);
        for (int i = 0; i < HEADERS.length; i++) {
            Cell cell = headerRow.createCell(i);
            cell.setCellValue(HEADERS[i]);
            cell.setCellStyle(styles.getHeaderStyle());
        }
    }

    private void populateDataRows(XSSFSheet sheet, List<ExcelRow> excelRows, StyleHelper styles) {
        int rowNum = 1;
        for (ExcelRow row : excelRows) {
            XSSFRow excelRow = sheet.createRow(rowNum++);
            populateRow(excelRow, row, styles);
        }
    }

    private void populateRow(XSSFRow excelRow, ExcelRow data, StyleHelper styles) {
        int col = 0;
        createCell(excelRow, col++, data.name(), styles.getCellStyle());
        createCell(excelRow, col++, data.version(), styles.getCellStyle());
        createNumericCell(excelRow, col++, data.accuracy(), styles.getNumberStyle());
        createNumericCell(excelRow, col++, data.precision(), styles.getNumberStyle());
        createNumericCell(excelRow, col++, data.recall(), styles.getNumberStyle());
        createNumericCell(excelRow, col++, data.macroF1(), styles.getNumberStyle());
        createNumericCell(excelRow, col++, data.microF1(), styles.getNumberStyle());
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
        for (int i = 0; i < HEADERS.length; i++) {
            sheet.autoSizeColumn(i);
        }
    }

    private void writeWorkbook(XSSFWorkbook workbook, String path) throws IOException {
        try (FileOutputStream fileOut = new FileOutputStream(path)) {
            workbook.write(fileOut);
        }
    }

    /**
     * Helper class encapsulating Excel cell styles used during export.
     *
     * <p>Style objects are created once per workbook and reused for header, text and numeric cells.</p>
     */
    @Getter
    private static class StyleHelper {
        private final XSSFCellStyle headerStyle;
        private final XSSFCellStyle cellStyle;
        private final XSSFCellStyle numberStyle;
        private final EvaluationExportProperties options;

        StyleHelper(XSSFWorkbook workbook, EvaluationExportProperties options) {
            if (options == null) {
                options = new EvaluationExportProperties();
            }
            this.options = options;
            this.headerStyle = createHeaderStyle(workbook);
            this.cellStyle = createCellStyle(workbook);
            this.numberStyle = createNumberStyle(workbook);
        }

        private XSSFCellStyle createHeaderStyle(XSSFWorkbook workbook) {
            XSSFFont headerFont = workbook.createFont();
            headerFont.setBold(true);
            headerFont.setFontHeightInPoints(options.getHeaderFontSize());

            XSSFCellStyle style = workbook.createCellStyle();
            style.setFont(headerFont);
            style.setAlignment(HorizontalAlignment.CENTER);
            style.setFillForegroundColor(IndexedColors.GREY_25_PERCENT.getIndex());
            style.setFillPattern(FillPatternType.SOLID_FOREGROUND);
            applyBorders(style);
            return style;
        }

        private XSSFCellStyle createCellStyle(XSSFWorkbook workbook) {
            XSSFCellStyle style = workbook.createCellStyle();
            applyBorders(style);
            return style;
        }

        private XSSFCellStyle createNumberStyle(XSSFWorkbook workbook) {
            XSSFCellStyle style = workbook.createCellStyle();
            applyBorders(style);
            XSSFDataFormat dataFormat = workbook.createDataFormat();
            style.setDataFormat(dataFormat.getFormat(options.getNumberFormat()));
            return style;
        }

        private void applyBorders(XSSFCellStyle style) {
            style.setBorderBottom(BorderStyle.THIN);
            style.setBorderTop(BorderStyle.THIN);
            style.setBorderLeft(BorderStyle.THIN);
            style.setBorderRight(BorderStyle.THIN);
        }
    }

    /**
     * Simple immutable DTO representing a single row in the exported Excel file.
     *
     * @param name      classification name
     * @param version   version identifier
     * @param accuracy  accuracy metric
     * @param precision precision metric
     * @param recall    recall metric
     * @param macroF1   macro F1 score
     * @param microF1   micro F1 score
     */
    @Builder
    private record ExcelRow(
            String name,
            String version,
            double accuracy,
            double precision,
            double recall,
            double macroF1,
            double microF1
    ) {}
}