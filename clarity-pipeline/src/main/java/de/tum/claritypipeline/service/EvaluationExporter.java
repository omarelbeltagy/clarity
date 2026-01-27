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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Service for exporting evaluation results and predictions from Neo4j to various formats.
 * <p>
 * This utility provides comprehensive export capabilities for classification experiment results,
 * supporting multiple output formats and use cases:
 * <ul>
 *   <li><b>Excel Export</b>: Aggregate evaluation metrics across all classification runs</li>
 *   <li><b>Prediction Export</b>: Individual predictions in competition-ready ZIP format</li>
 *   <li><b>Custom Evaluation</b>: Multi-label evaluation metrics for evasion-level analysis</li>
 * </ul>
 *
 * <h2>Excel Export Features</h2>
 * Generates formatted XLSX workbooks containing:
 * <ul>
 *   <li>Classification run names and versions</li>
 *   <li>Accuracy, Precision, Recall metrics</li>
 *   <li>Macro F1 and Micro F1 scores</li>
 *   <li>Configurable formatting (borders, number formats, header styles)</li>
 *   <li>Optional value rounding to specified decimal places</li>
 * </ul>
 *
 * <h2>Prediction Export Format</h2>
 * Creates ZIP archives containing:
 * <ul>
 *   <li>Plain text file named "prediction"</li>
 *   <li>One predicted label per line (in QA index order)</li>
 *   <li>Automatic label mapping if taxonomy mapping is enabled</li>
 *   <li>Compatible with competition submission formats</li>
 * </ul>
 *
 * <h2>Custom Evaluation Metrics</h2>
 * Supports advanced multi-label evaluation:
 * <ul>
 *   <li>Handles multiple annotators per QA (annotator1, annotator2, annotator3)</li>
 *   <li>Computes multi-label macro F1 for evasion-level analysis</li>
 *   <li>Filters out QAs with incomplete annotations</li>
 *   <li>Useful for fine-grained evaluation beyond single ground truth labels</li>
 * </ul>
 *
 * <h2>Factory Methods</h2>
 * The class provides multiple factory methods for initialization:
 * <ul>
 *   <li>{@link #create()}: Default Neo4j credentials and export options</li>
 *   <li>{@link #create(Neo4jCredentials)}: Custom credentials, default options</li>
 *   <li>{@link #create(EvaluationExportProperties)}: Default credentials, custom options</li>
 *   <li>{@link #create(Neo4jCredentials, EvaluationExportProperties)}: Full customization</li>
 *   <li>{@link #fromCredentialsFile(String)}: Load credentials from file</li>
 * </ul>
 *
 * <h2>Configuration Options</h2>
 * Export behavior is controlled via {@link EvaluationExportProperties}:
 * <ul>
 *   <li><b>sheetName</b>: Excel worksheet name</li>
 *   <li><b>roundToDigits</b>: Decimal places for metric rounding (0 = no rounding)</li>
 *   <li><b>headerFontSize</b>: Font size for Excel headers</li>
 *   <li><b>numberFormat</b>: Excel number format string (e.g., "0.0000")</li>
 * </ul>
 *
 * <h2>Error Handling</h2>
 * <ul>
 *   <li>Throws RuntimeException on critical Excel export failures</li>
 *   <li>Logs warnings for missing evaluations without stopping export</li>
 *   <li>Logs errors for individual evaluation computation failures</li>
 *   <li>Continues processing remaining runs when possible</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>
 * // Excel export with all runs
 * EvaluationExporter exporter = EvaluationExporter.create();
 * exporter.exportAsExcel("results/evaluation.xlsx");
 *
 * // Prediction export for specific run
 * exporter.exportResult("properties/gpt-5.1.yaml", "predictions/run1.zip");
 *
 * // Custom multi-label evaluation
 * exporter.generateCustomEvaluation("properties/gpt-5.1.yaml");
 * </pre>
 *
 * @see EvaluationExportProperties
 * @see ClassificationProperties
 * @see ModelEvaluator
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
     * Creates an exporter using default Neo4j connection settings and default export options.
     *
     * @return a new EvaluationExporter instance
     * @throws IOException if the default Neo4j client could not be initialized
     */
    public static EvaluationExporter create() throws IOException {
        return new EvaluationExporter(new Neo4jClient(), new EvaluationExportProperties());
    }

    /**
     * Creates an exporter with explicit Neo4j credentials and default export options.
     *
     * @param credentials Neo4j credentials to use for the client
     * @return configured EvaluationExporter
     */
    public static EvaluationExporter create(Neo4jCredentials credentials) {
        return new EvaluationExporter(new Neo4jClient(credentials), new EvaluationExportProperties());
    }

    /**
     * Creates an exporter using default Neo4j connection settings and the given options.
     *
     * @param options export formatting and behavior options (may be null for defaults)
     * @return configured EvaluationExporter
     * @throws IOException if the default Neo4j client could not be initialized
     */
    public static EvaluationExporter create(EvaluationExportProperties options) throws IOException {
        return new EvaluationExporter(new Neo4jClient(), options);
    }

    /**
     * Creates an exporter with explicit Neo4j credentials and given export options.
     *
     * @param credentials Neo4j credentials for the client
     * @param options     export formatting and behavior options
     * @return configured EvaluationExporter
     */
    public static EvaluationExporter create(Neo4jCredentials credentials, EvaluationExportProperties options) {
        return new EvaluationExporter(new Neo4jClient(credentials), options);
    }

    /**
     * Loads Neo4j credentials from a file and creates a new exporter.
     *
     * @param credentialsFile path to a YAML credentials file
     * @return configured EvaluationExporter
     * @throws IOException if the credentials file cannot be read
     */
    public static EvaluationExporter fromCredentialsFile(String credentialsFile) throws IOException {
        return create(Neo4jCredentials.load(credentialsFile));
    }

    /**
     * Exports aggregated evaluation metrics to an Excel workbook.
     * <p>
     * This method:
     * <ol>
     *   <li>Retrieves all Classification nodes from Neo4j</li>
     *   <li>Collects evaluation metrics from each classification run</li>
     *   <li>Formats data according to export options (rounding, styling)</li>
     *   <li>Generates an XLSX file with formatted metrics table</li>
     * </ol>
     * <p>
     * The resulting Excel file contains one row per classification version with columns:
     * <ul>
     *   <li>Name: Classification run name</li>
     *   <li>Version: Run version identifier</li>
     *   <li>Accuracy: Overall accuracy metric</li>
     *   <li>Precision: Weighted precision</li>
     *   <li>Recall: Weighted recall</li>
     *   <li>Macro F1: Unweighted average F1 score</li>
     *   <li>Micro F1: Weighted average F1 score</li>
     * </ul>
     *
     * @param path output file path for the generated Excel workbook (*.xlsx)
     * @throws RuntimeException if Excel generation or file writing fails
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

    /**
     * Generates custom multi-label evaluation metrics for a classification run.
     * <p>
     * This method performs advanced evaluation considering multiple annotators:
     * <ol>
     *   <li>Retrieves all classification results for the specified run</li>
     *   <li>Fetches corresponding QAs with ground truth and annotator labels</li>
     *   <li>Filters QAs with complete annotation data (all 3 annotators)</li>
     *   <li>Computes multi-label macro F1 using {@link ModelEvaluator}</li>
     * </ol>
     * <p>
     * Useful for evasion-level evaluation where multiple valid interpretations exist.
     *
     * @param classificationPropertiesPath path to classification properties YAML file
     * @throws IOException if properties file cannot be loaded
     */
    public void generateCustomEvaluation(String classificationPropertiesPath) throws IOException {
        ClassificationProperties classificationProperties = ClassificationProperties.load(classificationPropertiesPath);
        generateCustomEvaluation(classificationProperties);
    }

    /**
     * Generates custom multi-label evaluation metrics for a classification run.
     * <p>
     * Overloaded version accepting a ClassificationProperties object directly.
     *
     * @param properties the classification properties object
     * @see #generateCustomEvaluation(String)
     */
    public void generateCustomEvaluation(ClassificationProperties properties) {
        log.info("Generating evaluation for classification run {} of {}", properties.getVersion(),
                 properties.getName());
        String query = String.format(
                """
                        MATCH (n:%s)--(cr:%s)--(c:%s)
                        WHERE cr.version = '%s'
                        AND c.name = '%s'
                        RETURN n
                        """,
                Neo4jNode.getLabel(ClassificationResult.class),
                Neo4jNode.getLabel(ClassificationProperties.class),
                Neo4jNode.getLabel(ClassificationProperties.Classification.class),
                properties.getVersion(),
                properties.getClassification().getName()
        );

        List<ClassificationResult> results = client.executeQuery(query, ClassificationResult.class);
        log.info("Found {} classification results for evaluation", results.size());
        List<List<String>> predictionsAndExpected =
                results.parallelStream()
                       .map(result -> {
                           String findQAQuery = String.format(
                                   """
                                           MATCH (cr:%s)--(n:%s)
                                           WHERE elementId(cr) = '%s'
                                           RETURN n
                                           """,
                                   Neo4jNode.getLabel(ClassificationResult.class),
                                   Neo4jNode.getLabel(QA.class),
                                   result.getElementId()
                           );

                           QA qa = client.executeQuery(findQAQuery, QA.class)
                                         .stream()
                                         .findFirst()
                                         .orElse(null);

                           if (qa == null) {
                               log.warn("No QA found for classification result {}. Could not generate evaluation",
                                        result.getElementId());
                               return null;
                           }
                           List<String> returnList = new ArrayList<>();
                           returnList.add(result.getName());
                           returnList.add(qa.getClarityLabel());
                           returnList.add(qa.getAnnotator1());
                           returnList.add(qa.getAnnotator2());
                           returnList.add(qa.getAnnotator3());
                           return returnList.contains(null) ? null : returnList;
                       })
                       .filter(Objects::nonNull)
                       .toList();

        List<String> predictions = predictionsAndExpected.stream()
                                                         .map(l -> l.get(0))
                                                         .toList();

        List<String> expected = predictionsAndExpected.stream()
                                                      .map(l -> l.get(1))
                                                      .toList();

        List<List<String>> annotations = predictionsAndExpected.stream()
                                                               .map(l -> l.subList(2, 5))
                                                               .toList();

        List<String> labels = properties.getTaxonomy().getCategories()
                                        .stream()
                                        .map(Taxonomy.Category::getName)
                                        .toList();

        try {
            ModelEvaluator evaluator = new ModelEvaluator(labels, predictions, expected);
            /*log.info("Evaluation Results (clarity level):");
            double accuracy = evaluator.accuracy();
            log.info("Accuracy: {}", String.format("%.2f", accuracy * 100));
            double precision = evaluator.precision();
            log.info("Precision: {}", String.format("%.2f", precision * 100));
            double recall = evaluator.recall();
            log.info("Recall: {}", String.format("%.2f", recall * 100));
            double microF1 = evaluator.microF1();
            log.info("Micro F1 Score: {}", String.format("%.2f", microF1 * 100));
            double clarityMacroF1 = evaluator.macroF1();
            log.info("Macro F1 Score: {}", String.format("%.2f", clarityMacroF1 * 100));*/
            double evasionMacroF1 = evaluator.multiLabelMacroF1(annotations);
            log.info("Evaluation Results (evasion level):");
            log.info("Macro F1 Score: {}", String.format("%.2f", evasionMacroF1 * 100));
        } catch (Exception e) {
            log.error("Error while evaluating classification run {}", properties.getVersion(), e);
        }

    }

    /**
     * Exports classification predictions to a ZIP file.
     * <p>
     * This method:
     * <ol>
     *   <li>Retrieves all classification results for the specified run (ordered by QA index)</li>
     *   <li>Maps predicted labels using taxonomy mapping if enabled</li>
     *   <li>Writes predictions to a temporary text file (one label per line)</li>
     *   <li>Creates a ZIP archive containing the prediction file</li>
     *   <li>Cleans up temporary files</li>
     * </ol>
     * <p>
     * The ZIP format is designed for competition submissions and contains a single
     * file named "prediction" with newline-separated labels.
     *
     * @param classificationPropertiesPath path to classification properties YAML file
     * @param outputFile                   path for the output ZIP file
     * @throws IOException      if file operations fail
     * @throws RuntimeException if required taxonomy categories or mappings are missing
     */
    public void exportResult(String classificationPropertiesPath, String outputFile) throws IOException {
        ClassificationProperties classificationProperties = ClassificationProperties.load(classificationPropertiesPath);
        exportResult(classificationProperties, outputFile);
    }

    /**
     * Exports classification predictions to a ZIP file.
     * <p>
     * Overloaded version accepting a ClassificationProperties object directly.
     *
     * @param classificationProperties the classification properties object
     * @param outputFile               path for the output ZIP file
     * @throws IOException if file operations fail
     * @see #exportResult(String, String)
     */
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
                        throw new RuntimeException(
                                "Could not find mapping for category with name " + result.getName());
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
     * Collects evaluation data from classification nodes and converts to Excel row DTOs.
     * <p>
     * Iterates through all classification runs, retrieves their evaluations,
     * and formats them for spreadsheet output.
     *
     * @param classifications list of Classification nodes from Neo4j
     * @return list of ExcelRow objects ready for Excel export
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
     * Retrieves the Evaluation object from a version node via reflection.
     * <p>
     * Uses reflection to call getEvaluation(Neo4jClient) on version objects,
     * enabling generic handling of different version node types.
     *
     * @param version version node object
     * @return Evaluation instance if available, null otherwise
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
                       .evasionMacroF1(evaluation.getEvasionMacroF1())
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
                                                  .evasionMacroF1(round(eval.getEvasionMacroF1(), factor))
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
        createNumericCell(excelRow, col++, data.evasionMacroF1(), styles.getNumberStyle());
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
     * Helper class for managing Excel cell styles during export.
     * <p>
     * Creates and caches reusable cell styles for:
     * <ul>
     *   <li><b>Header Style</b>: Bold font, centered, gray background</li>
     *   <li><b>Cell Style</b>: Standard text cells with borders</li>
     *   <li><b>Number Style</b>: Numeric cells with custom formatting</li>
     * </ul>
     * <p>
     * All styles include thin borders on all sides for a clean table appearance.
     */
    @Getter
    private static class StyleHelper {
        private final XSSFCellStyle headerStyle;
        private final XSSFCellStyle cellStyle;
        private final XSSFCellStyle numberStyle;
        private final EvaluationExportProperties options;

        /**
         * Constructs a StyleHelper with styles configured according to export options.
         *
         * @param workbook the Excel workbook for style creation
         * @param options  export options controlling formatting
         */
        StyleHelper(XSSFWorkbook workbook, EvaluationExportProperties options) {
            if (options == null) {
                options = new EvaluationExportProperties();
            }
            this.options = options;
            this.headerStyle = createHeaderStyle(workbook);
            this.cellStyle = createCellStyle(workbook);
            this.numberStyle = createNumberStyle(workbook);
        }

        /**
         * Creates the header row style with bold font and gray background.
         *
         * @param workbook the Excel workbook
         * @return configured header cell style
         */
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

        /**
         * Creates the standard cell style for text content.
         *
         * @param workbook the Excel workbook
         * @return configured cell style with borders
         */
        private XSSFCellStyle createCellStyle(XSSFWorkbook workbook) {
            XSSFCellStyle style = workbook.createCellStyle();
            applyBorders(style);
            return style;
        }

        /**
         * Creates the numeric cell style with custom number formatting.
         *
         * @param workbook the Excel workbook
         * @return configured numeric cell style
         */
        private XSSFCellStyle createNumberStyle(XSSFWorkbook workbook) {
            XSSFCellStyle style = workbook.createCellStyle();
            applyBorders(style);
            XSSFDataFormat dataFormat = workbook.createDataFormat();
            style.setDataFormat(dataFormat.getFormat(options.getNumberFormat()));
            return style;
        }

        /**
         * Applies thin borders to all four sides of a cell style.
         *
         * @param style the cell style to modify
         */
        private void applyBorders(XSSFCellStyle style) {
            style.setBorderBottom(BorderStyle.THIN);
            style.setBorderTop(BorderStyle.THIN);
            style.setBorderLeft(BorderStyle.THIN);
            style.setBorderRight(BorderStyle.THIN);
        }
    }

    /**
     * Immutable record representing a single row in the exported Excel evaluation file.
     * <p>
     * Bundles together all evaluation metrics for a single classification run version.
     *
     * @param name           classification run name
     * @param version        version identifier string
     * @param accuracy       overall classification accuracy (0.0 to 1.0)
     * @param precision      weighted precision metric (0.0 to 1.0)
     * @param recall         weighted recall metric (0.0 to 1.0)
     * @param macroF1        macro-averaged F1 score (0.0 to 1.0)
     * @param microF1        micro-averaged F1 score (0.0 to 1.0)
     * @param evasionMacroF1 multi-label macro F1 score for evasion-level evaluation (0.0 to 1.0)
     */
    @Builder
    private record ExcelRow(
            String name,
            String version,
            double accuracy,
            double precision,
            double recall,
            double macroF1,
            double microF1,
            double evasionMacroF1
    ) {}
}

