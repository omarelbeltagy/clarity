package de.tum.claritypipeline.service;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import de.tum.claritypipeline.model.Cluster;
import de.tum.claritypipeline.model.Evaluation;
import de.tum.claritypipeline.model.properties.EvaluationExportOptions;
import lombok.Builder;
import lombok.Getter;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Utility responsible for exporting evaluation results from Neo4j into an Excel file.
 *
 * <p>This class queries the Neo4j database for Cluster / Classification / Version nodes,
 * retrieves Evaluation objects from version nodes and writes a tabular XLSX file with
 * configurable formatting options.</p>
 *
 * <p>Creation is performed via factory methods which allow providing Neo4j credentials
 * or custom EvaluationExportOptions.</p>
 */
public class EvaluationExporter {
    private static final Logger log = LoggerFactory.getLogger(EvaluationExporter.class);

    private static final String[] HEADERS = {
            "Cluster", "Name", "Version", "Accuracy",
            "Precision", "Recall", "Macro F1", "Micro F1"
    };

    private final Neo4jClient client;
    private final EvaluationExportOptions options;

    private EvaluationExporter(Neo4jClient client, EvaluationExportOptions options) {
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
        return new EvaluationExporter(new Neo4jClient(), new EvaluationExportOptions());
    }

    /**
     * Create a new exporter with explicit Neo4j credentials and default export options.
     *
     * @param credentials Neo4j credentials to use for the client
     * @return configured EvaluationExporter
     */
    public static EvaluationExporter create(Neo4jCredentials credentials) {
        return new EvaluationExporter(new Neo4jClient(credentials), new EvaluationExportOptions());
    }

    /**
     * Create a new exporter using default Neo4j connection settings and the given options.
     *
     * @param options export formatting and behavior options (may be null)
     * @return configured EvaluationExporter
     * @throws IOException if the default Neo4j client could not be initialized
     */
    public static EvaluationExporter create(EvaluationExportOptions options) throws IOException {
        return new EvaluationExporter(new Neo4jClient(), options);
    }

    /**
     * Create a new exporter with explicit Neo4j credentials and given export options.
     *
     * @param credentials Neo4j credentials for the client
     * @param options     export formatting and behavior options
     * @return configured EvaluationExporter
     */
    public static EvaluationExporter create(Neo4jCredentials credentials, EvaluationExportOptions options) {
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
     * <p>This method collects clusters and their child nodes, extracts Evaluation objects
     * from version nodes and writes the results to the given file path. On failure a
     * RuntimeException is thrown.</p>
     *
     * @param path output path for the generated Excel file (XLSX)
     */
    public void exportAsExcel(String path) {
        try {
            List<Cluster> clusters = client.findNodes(Map.of(), Cluster.class);
            List<ExcelRow> excelRows = readExcelRows(clusters);
            exportToExcelFile(excelRows, path);
        } catch (Exception e) {
            log.error("Failed to export evaluation data to Excel", e);
            throw new RuntimeException("Excel export failed", e);
        }
    }

    /**
     * Read the Evaluation objects from the provided clusters and map them to ExcelRow DTOs.
     *
     * @param clusters list of Cluster nodes retrieved from Neo4j
     * @return list of ExcelRow entries to be written to the spreadsheet
     */
    private List<ExcelRow> readExcelRows(List<Cluster> clusters) {
        return clusters.stream()
                       .flatMap(cluster -> cluster.getChildren(client).stream()
                                                  .flatMap(classification -> classification.getChildren(client).stream()
                                                                                           .map(version -> createExcelRow(
                                                                                                   cluster,
                                                                                                   classification,
                                                                                                   version))))
                       .filter(Objects::nonNull)
                       .toList();
    }

    /**
     * Attempt to retrieve an Evaluation instance from a version node via reflection.
     *
     * @param version version node object from which to fetch the Evaluation
     * @return Evaluation instance if available, otherwise null
     */
    private Evaluation getEvaluation(Object version) {
        try {
            return (Evaluation) version.getClass()
                                       .getMethod("getEvaluation", Neo4jClient.class)
                                       .invoke(version, client);
        } catch (Exception e) {
            log.warn("Failed to get evaluation for version", e);
            return null;
        }
    }

    private ExcelRow createExcelRow(Cluster cluster, Object classification, Object version) {
        Evaluation evaluation = getEvaluation(version);
        if (evaluation == null) {
            return null;
        }

        if (shouldRoundValues()) {
            evaluation = roundEvaluation(evaluation);
        }

        return ExcelRow.builder()
                       .cluster(cluster.getName())
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

    private Evaluation roundEvaluation(Evaluation eval) {
        double factor = Math.pow(10, options.getRoundToDigits());
        return Evaluation.builder()
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
        createCell(excelRow, col++, data.cluster(), styles.getCellStyle());
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
        private final EvaluationExportOptions options;

        StyleHelper(XSSFWorkbook workbook, EvaluationExportOptions options) {
            if (options == null) {
                options = new EvaluationExportOptions();
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
     * @param cluster cluster name
     * @param name classification name
     * @param version version identifier
     * @param accuracy accuracy metric
     * @param precision precision metric
     * @param recall recall metric
     * @param macroF1 macro F1 score
     * @param microF1 micro F1 score
     */
    @Builder
    private record ExcelRow(
            String cluster,
            String name,
            String version,
            double accuracy,
            double precision,
            double recall,
            double macroF1,
            double microF1
    ) {}
}