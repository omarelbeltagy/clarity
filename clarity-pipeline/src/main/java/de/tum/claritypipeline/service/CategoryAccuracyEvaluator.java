package de.tum.claritypipeline.service;

import de.tum.clarityneo4j.core.Neo4jClient;
import de.tum.clarityneo4j.model.Neo4jCredentials;
import lombok.Builder;
import lombok.Data;
import lombok.Value;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Evaluates and exports per-category classification accuracy for a list of classification runs.
 *
 * <p>For each run, two accuracy breakdowns are computed:
 * <ul>
 *   <li><b>Evasion accuracy</b>: per predicted evasion category (9 fine-grained labels).
 *       A prediction is considered correct if it matches at least one of the three annotator labels
 *       ({@code annotator1}, {@code annotator2}, {@code annotator3}) stored on the QA node.</li>
 *   <li><b>Clarity accuracy</b>: per predicted clarity category (3 coarse labels).
 *       The predicted evasion label is first mapped to its corresponding clarity label via the
 *       fixed taxonomy mapping, and then compared against {@code qa.clarityLabel}.</li>
 * </ul>
 *
 * <p>Results are written to an Excel file ({@code .xlsx}), with one sheet per classification run.
 * Sheet names are derived from the run's name and version and are sanitized to conform to Excel's
 * 31-character sheet name limit.
 *
 * <p>Usage example:
 * <pre>{@code
 * CategoryAccuracyEvaluator evaluator = CategoryAccuracyEvaluator.create();
 * evaluator.exportCategoryAccuracies(
 *     CategoryAccuracyEvaluator.CategoryEvaluationOptions.builder()
 *         .classificationRuns(List.of(
 *             CategoryAccuracyEvaluator.CategoryEvaluationOptions.ClassificationRun.builder()
 *                 .classificationName("GPT 5.2")
 *                 .classificationVersion("evasion-based:single:few-shot:v1")
 *                 .build()
 *         ))
 *         .excelPath("/output/category_accuracy.xlsx")
 *         .build()
 * );
 * }</pre>
 */
public class CategoryAccuracyEvaluator {
    
    private static final Logger log = LoggerFactory.getLogger(CategoryAccuracyEvaluator.class);
    
    /** Canonical row order for the evasion-accuracy pivot sheet. */
    private static final List<String> EVASION_CATEGORY_ORDER = List.of(
            "Explicit", "Implicit", "General", "Partial/half-answer", "Dodging",
            "Deflection", "Declining to answer", "Claims ignorance", "Clarification"
    );
    
    /** Canonical row order for the clarity-accuracy pivot sheet. */
    private static final List<String> CLARITY_CATEGORY_ORDER = List.of(
            "Clear Reply", "Ambivalent", "Clear Non-Reply"
    );
    
    private final Neo4jClient client;
    
    /**
     * Creates a new evaluator using the provided {@link Neo4jClient}.
     *
     * @param client the Neo4j client used to execute queries
     */
    private CategoryAccuracyEvaluator(Neo4jClient client) {
        this.client = client;
    }
    
    /**
     * Creates a new evaluator by loading Neo4j credentials from the default configuration file.
     *
     * @throws IOException if the credentials file cannot be read
     */
    private CategoryAccuracyEvaluator() throws IOException {
        this.client = new Neo4jClient();
    }
    
    /**
     * Creates a new evaluator using the provided Neo4j credentials.
     *
     * @param neo4jCredentials the credentials used to connect to the Neo4j instance
     */
    private CategoryAccuracyEvaluator(Neo4jCredentials neo4jCredentials) {
        this.client = new Neo4jClient(neo4jCredentials);
    }
    
    // -------------------------------------------------------------------------
    // Static factory methods
    // -------------------------------------------------------------------------
    
    /**
     * Creates a {@link CategoryAccuracyEvaluator} wrapping the given {@link Neo4jClient}.
     *
     * @param client an already-connected Neo4j client
     * @return a new evaluator instance
     */
    public static CategoryAccuracyEvaluator of(Neo4jClient client) {
        return new CategoryAccuracyEvaluator(client);
    }
    
    /**
     * Creates a {@link CategoryAccuracyEvaluator} using the supplied Neo4j credentials.
     *
     * @param credentials credentials for the Neo4j connection
     * @return a new evaluator instance
     */
    public static CategoryAccuracyEvaluator of(Neo4jCredentials credentials) {
        return new CategoryAccuracyEvaluator(credentials);
    }
    
    /**
     * Creates a {@link CategoryAccuracyEvaluator} by loading Neo4j credentials from the default configuration file on
     * the classpath.
     *
     * @return a new evaluator instance
     * @throws IOException if the configuration file cannot be read
     */
    public static CategoryAccuracyEvaluator create() throws IOException {
        return new CategoryAccuracyEvaluator();
    }
    
    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------
    
    /**
     * Exports per-category accuracy for a list of classification runs to an Excel file with two sheets:
     * <ol>
     *   <li><b>Evasion Accuracy</b> – rows are the nine evasion technique categories in taxonomy order;
     *       columns are the individual classification runs. Each cell contains the accuracy percentage
     *       for that category in that run (correct = at least one annotator agrees).</li>
     *   <li><b>Clarity Accuracy</b> – same pivot layout for the three clarity labels, where predictions
     *       are mapped from evasion categories via the hardcoded taxonomy mapping.</li>
     * </ol>
     * Both sheets share a "Total" column (number of test-set predictions for that category, taken from
     * the first run that has data for the category) and one accuracy column per run.
     *
     * <p>The Excel file is written to the path specified in {@link CategoryEvaluationOptions#getExcelPath()}.
     *
     * @param options configuration specifying the classification runs to evaluate and the output path
     * @throws IOException if the Excel file cannot be written
     */
    public void exportCategoryAccuracies(CategoryEvaluationOptions options) throws IOException {
        // Collect all results upfront so we can build the pivot tables
        List<CategoryAccuracyResult> allResults = new ArrayList<>();
        for (CategoryEvaluationOptions.ClassificationRun run : options.getClassificationRuns()) {
            log.info(
                    "Querying classification {}({})",
                    run.getClassificationName(), run.getClassificationVersion()
            );
            allResults.add(getResults(run.getClassificationName(), run.getClassificationVersion()));
        }
        
        try (XSSFWorkbook workbook = new XSSFWorkbook()) {
            StyleHelper styles = new StyleHelper(workbook);
            int columnCount = 2 + allResults.size();
            
            XSSFSheet evasionSheet = workbook.createSheet("Evasion Accuracy");
            writePivotSheet(
                    evasionSheet, allResults,
                    CategoryAccuracyResult::getEvasionRows, EVASION_CATEGORY_ORDER, styles
            );
            autoSizeColumns(evasionSheet, columnCount);
            
            XSSFSheet claritySheet = workbook.createSheet("Clarity Accuracy");
            writePivotSheet(
                    claritySheet, allResults,
                    CategoryAccuracyResult::getClarityRows, CLARITY_CATEGORY_ORDER, styles
            );
            autoSizeColumns(claritySheet, columnCount);
            
            writeWorkbook(workbook, options.getExcelPath());
        }
        log.info("Exported category accuracies to {}", options.getExcelPath());
    }
    
    // -------------------------------------------------------------------------
    // Query execution
    // -------------------------------------------------------------------------
    
    /**
     * Retrieves per-category accuracy for a single classification run by executing two Cypher aggregation queries
     * against the Neo4j graph.
     *
     * <p><b>Evasion query:</b> groups predicted labels by {@code ClassificationResult.name} and
     * counts predictions that match at least one of the three annotator labels on the QA node.
     *
     * <p><b>Clarity query:</b> maps each predicted evasion label to its clarity label via the
     * hardcoded taxonomy mapping and groups by the mapped clarity label, comparing against {@code QA.clarityLabel}.
     *
     * <p>Note: this method only supports evasion-based classification runs where the predicted label
     * is one of the nine fine-grained evasion technique names. Direct clarity classification runs (where the predicted
     * label is already a clarity label) are not supported.
     *
     * @param classificationName    the {@code name} property of the {@code ClassificationProperties} node
     * @param classificationVersion the {@code version} property of the {@code ClassificationProperties} node
     * @return a {@link CategoryAccuracyResult} containing evasion and clarity accuracy rows
     */
    private CategoryAccuracyResult getResults(String classificationName, String classificationVersion) {
        log.info("Getting stats for classification {}({})", classificationName, classificationVersion);
        
        String evasionQuery = """
                              MATCH (qa:QA)-[:HAS_CLASSIFICATION]->(cr:ClassificationResult)-[:GENERATED_BY]->(cp:ClassificationProperties)
                              WHERE cp.name = '%s'
                                AND cp.version = '%s'
                                AND qa.test = true
                              RETURN
                                cr.name                                                                          AS predicted_category,
                                COUNT(*)                                                                         AS total,
                                SUM(CASE WHEN cr.name IN [qa.annotator1, qa.annotator2, qa.annotator3]
                                         THEN 1 ELSE 0 END)                                                     AS correct,
                                ROUND(100.0 * SUM(CASE WHEN cr.name IN [qa.annotator1, qa.annotator2, qa.annotator3]
                                                       THEN 1 ELSE 0 END) / COUNT(*), 1)                       AS accuracy_pct
                              ORDER BY total DESC
                              """.formatted(classificationName, classificationVersion);
        
        String clarityQuery = """
                              MATCH (qa:QA)-[:HAS_CLASSIFICATION]->(cr:ClassificationResult)-[:GENERATED_BY]->(cp:ClassificationProperties)
                              WHERE cp.name = '%s'
                                AND cp.version = '%s'
                                AND qa.test = true
                              WITH qa,
                                CASE cr.name
                                  WHEN 'Explicit'            THEN 'Clear Reply'
                                  WHEN 'Implicit'            THEN 'Ambivalent'
                                  WHEN 'General'             THEN 'Ambivalent'
                                  WHEN 'Partial/half-answer' THEN 'Ambivalent'
                                  WHEN 'Dodging'             THEN 'Ambivalent'
                                  WHEN 'Deflection'          THEN 'Ambivalent'
                                  WHEN 'Declining to answer' THEN 'Clear Non-Reply'
                                  WHEN 'Claims ignorance'    THEN 'Clear Non-Reply'
                                  WHEN 'Clarification'       THEN 'Clear Non-Reply'
                                END AS mapped_clarity
                              RETURN
                                mapped_clarity                                                                   AS predicted_clarity,
                                COUNT(*)                                                                         AS total,
                                SUM(CASE WHEN mapped_clarity = qa.clarityLabel THEN 1 ELSE 0 END)              AS correct,
                                ROUND(100.0 * SUM(CASE WHEN mapped_clarity = qa.clarityLabel
                                                       THEN 1 ELSE 0 END) / COUNT(*), 1)                       AS accuracy_pct
                              ORDER BY total DESC
                              """.formatted(classificationName, classificationVersion);
        
        List<CategoryAccuracyRow> evasionRows = client.getRecords(evasionQuery)
                .map(record -> new CategoryAccuracyRow(
                        record.get("predicted_category").asString("(null)"),
                        record.get("total").asLong(),
                        record.get("correct").asLong(),
                        record.get("accuracy_pct").asDouble()
                ))
                .collect(Collectors.toList());
        
        List<CategoryAccuracyRow> clarityRows = client.getRecords(clarityQuery)
                .map(record -> new CategoryAccuracyRow(
                        record.get("predicted_clarity").asString("(null)"),
                        record.get("total").asLong(),
                        record.get("correct").asLong(),
                        record.get("accuracy_pct").asDouble()
                ))
                .collect(Collectors.toList());
        
        return new CategoryAccuracyResult(classificationName, classificationVersion, evasionRows, clarityRows);
    }
    
    // -------------------------------------------------------------------------
    // Excel writing helpers
    // -------------------------------------------------------------------------
    
    /**
     * Writes a pivot table into the given sheet.
     *
     * <p>Layout:
     * <pre>
     * Category | Total | Run-1 label | Run-2 label | …
     * Cat A    | 105   | 44.8        | 46.2        | …
     * Cat B    | 72    | 88.9        | 90.1        | …
     * </pre>
     * Rows follow the order defined by {@code categoryOrder}. The "Total" column is taken from the first run that
     * returns data for that category (the test set is fixed, so totals should be identical across runs). If a run has
     * no predictions for a category the accuracy cell is left blank. Each run column header shows
     * {@code "name / version"}.
     *
     * @param sheet         the sheet to write into
     * @param allResults    results for every classification run, in the desired column order
     * @param rowsExtractor selects either {@code evasionRows} or {@code clarityRows} from a result
     * @param categoryOrder canonical row ordering for the categories in this sheet
     * @param styles        pre-built cell styles for the workbook
     */
    private void writePivotSheet(
            XSSFSheet sheet,
            List<CategoryAccuracyResult> allResults,
            Function<CategoryAccuracyResult, List<CategoryAccuracyRow>> rowsExtractor,
            List<String> categoryOrder,
            StyleHelper styles
    ) {
        
        int rowNum = 0;
        
        // Header row
        XSSFRow headerRow = sheet.createRow(rowNum++);
        createCell(headerRow, 0, "Category", styles.getHeaderStyle());
        createCell(headerRow, 1, "Total", styles.getHeaderStyle());
        for (int i = 0; i < allResults.size(); i++) {
            CategoryAccuracyResult r = allResults.get(i);
            createCell(
                    headerRow, 2 + i,
                    r.getClassificationName() + " / " + r.getClassificationVersion(),
                    styles.getHeaderStyle()
            );
        }
        
        // One data row per category in canonical taxonomy order
        for (String category : categoryOrder) {
            XSSFRow dataRow = sheet.createRow(rowNum++);
            createCell(dataRow, 0, category, styles.getCellStyle());
            
            // Total: first run that has data for this category
            long total = allResults.stream()
                    .flatMap(r -> rowsExtractor.apply(r).stream())
                    .filter(row -> category.equals(row.getCategory()))
                    .mapToLong(CategoryAccuracyRow::getTotal)
                    .findFirst()
                    .orElse(0L);
            createNumericCell(dataRow, 1, total, styles.getNumberStyle());
            
            // Accuracy per run
            for (int i = 0; i < allResults.size(); i++) {
                int finalI = i;
                rowsExtractor.apply(allResults.get(i)).stream()
                        .filter(row -> category.equals(row.getCategory()))
                        .findFirst()
                        .ifPresentOrElse(
                                row -> createNumericCell(
                                        dataRow, 2 + finalI, row.getAccuracyPct(), styles.getNumberStyle()),
                                () -> createCell(dataRow, 2 + finalI, "-", styles.getCellStyle())
                        );
            }
        }
    }
    
    /**
     * Creates a string cell at the given column position with the supplied style.
     *
     * @param row    the row to add the cell to
     * @param column zero-based column index
     * @param value  the string value; {@code null} is written as an empty string
     * @param style  the cell style to apply
     */
    private void createCell(XSSFRow row, int column, String value, XSSFCellStyle style) {
        Cell cell = row.createCell(column);
        cell.setCellValue(value != null ? value : "");
        cell.setCellStyle(style);
    }
    
    /**
     * Creates a numeric cell for a {@code double} value at the given column position.
     *
     * @param row    the row to add the cell to
     * @param column zero-based column index
     * @param value  the numeric value
     * @param style  the cell style to apply
     */
    private void createNumericCell(XSSFRow row, int column, double value, XSSFCellStyle style) {
        Cell cell = row.createCell(column);
        cell.setCellValue(value);
        cell.setCellStyle(style);
    }
    
    /**
     * Creates a numeric cell for a {@code long} value at the given column position.
     *
     * @param row    the row to add the cell to
     * @param column zero-based column index
     * @param value  the numeric value
     * @param style  the cell style to apply
     */
    private void createNumericCell(XSSFRow row, int column, long value, XSSFCellStyle style) {
        Cell cell = row.createCell(column);
        cell.setCellValue(value);
        cell.setCellStyle(style);
    }
    
    /**
     * Auto-sizes the first {@code columnCount} columns of the given sheet so their contents are fully visible without
     * manual resizing.
     *
     * @param sheet       the sheet whose columns should be resized
     * @param columnCount the number of columns (starting at index 0) to auto-size
     */
    private void autoSizeColumns(XSSFSheet sheet, int columnCount) {
        for (int i = 0; i < columnCount; i++) {
            sheet.autoSizeColumn(i);
        }
    }
    
    /**
     * Sanitizes a string so it can be used as an Excel sheet name. Excel sheet names must not exceed 31 characters and
     * must not contain the characters {@code \ / * ? : [ ]}.
     *
     * @param name the raw name to sanitize
     * @return a sanitized sheet name of at most 31 characters
     */
    private String sanitizeSheetName(String name) {
        String sanitized = name.replaceAll("[\\\\/*?:\\[\\]]", "_");
        return sanitized.length() > 31 ? sanitized.substring(0, 31) : sanitized;
    }
    
    /**
     * Returns a sheet name that is both Excel-safe and unique within the given workbook.
     *
     * <p>The name is first sanitized via {@link #sanitizeSheetName(String)}. If the resulting
     * name already exists in the workbook, a numeric suffix ({@code _1}, {@code _2}, …) is appended (and the base is
     * truncated accordingly) until the name is unique.
     *
     * @param workbook the workbook to check for existing sheet names
     * @param name     the desired sheet name (before sanitization)
     * @return a sanitized, unique sheet name of at most 31 characters
     */
    private String uniqueSheetName(XSSFWorkbook workbook, String name) {
        String base = sanitizeSheetName(name);
        if (workbook.getSheet(base) == null) {
            return base;
        }
        for (int i = 1; ; i++) {
            String suffix = "_" + i;
            String candidate = base.length() + suffix.length() > 31
                    ? base.substring(0, 31 - suffix.length()) + suffix
                    : base + suffix;
            if (workbook.getSheet(candidate) == null) {
                return candidate;
            }
        }
    }
    
    /**
     * Writes the given workbook to disk at the specified path.
     *
     * @param workbook the workbook to write
     * @param path     the target file path (created or overwritten)
     * @throws IOException if the file cannot be written
     */
    private void writeWorkbook(XSSFWorkbook workbook, String path) throws IOException {
        try (FileOutputStream fileOut = new FileOutputStream(path)) {
            workbook.write(fileOut);
        }
    }
    
    // -------------------------------------------------------------------------
    // Inner data classes
    // -------------------------------------------------------------------------
    
    /**
     * A single row returned by a per-category accuracy query, holding the predicted label, the total prediction count,
     * the number of correct predictions, and the accuracy percentage.
     */
    @Value
    private static class CategoryAccuracyRow {
        
        /** The predicted label (evasion category name or clarity label). */
        String category;
        
        /** Total number of test-set predictions for this label. */
        long total;
        
        /** Number of correct predictions within the {@code total}. */
        long correct;
        
        /** Accuracy as a percentage rounded to one decimal place (0–100). */
        double accuracyPct;
    }
    
    /**
     * Aggregated accuracy results for a single classification run, containing separate lists of accuracy rows for the
     * evasion and clarity dimensions.
     */
    @Value
    private static class CategoryAccuracyResult {
        
        /** The {@code name} property of the corresponding {@code ClassificationProperties} node. */
        String classificationName;
        
        /** The {@code version} property of the corresponding {@code ClassificationProperties} node. */
        String classificationVersion;
        
        /** Per-evasion-category accuracy rows, ordered by total predictions descending. */
        List<CategoryAccuracyRow> evasionRows;
        
        /** Per-clarity-category accuracy rows, ordered by total predictions descending. */
        List<CategoryAccuracyRow> clarityRows;
    }
    
    /**
     * Configuration for a batch export of category accuracies.
     *
     * <p>Build instances via the Lombok {@code @Builder}:
     * <pre>{@code
     * CategoryEvaluationOptions options = CategoryEvaluationOptions.builder()
     *     .classificationRuns(List.of(...))
     *     .excelPath("/output/results.xlsx")
     *     .build();
     * }</pre>
     */
    @Data
    @Builder
    public static class CategoryEvaluationOptions {
        
        /**
         * The ordered list of classification runs to evaluate. Each run produces one sheet in the output workbook.
         */
        private List<ClassificationRun> classificationRuns;
        
        /**
         * Absolute or relative path to the target {@code .xlsx} file. The file is created or overwritten when the
         * export runs.
         */
        private String excelPath;
        
        /**
         * Identifies a single classification run by its {@code ClassificationProperties} name and version.
         *
         * <p>Build instances via the Lombok {@code @Builder}:
         * <pre>{@code
         * ClassificationRun run = ClassificationRun.builder()
         *     .classificationName("GPT 5.2")
         *     .classificationVersion("evasion-based:single:few-shot:v1")
         *     .build();
         * }</pre>
         */
        @Data
        @Builder
        public static class ClassificationRun {
            
            /**
             * The {@code name} property of the {@code ClassificationProperties} node in Neo4j. Corresponds to the
             * {@code name} field in the run's YAML configuration file.
             */
            private String classificationName;
            
            /**
             * The {@code version} property of the {@code ClassificationProperties} node in Neo4j. Corresponds to the
             * {@code version} field in the run's YAML configuration file.
             */
            private String classificationVersion;
        }
    }
    
    // -------------------------------------------------------------------------
    // Style helper
    // -------------------------------------------------------------------------
    
    /**
     * Creates and caches all {@link XSSFCellStyle} instances needed by the exporter. Styles must be created once per
     * workbook and reused across cells to avoid exceeding Excel's internal style limit.
     */
    private static class StyleHelper {
        
        private final XSSFCellStyle titleStyle;
        
        private final XSSFCellStyle sectionStyle;
        
        private final XSSFCellStyle headerStyle;
        
        private final XSSFCellStyle cellStyle;
        
        private final XSSFCellStyle numberStyle;
        
        /**
         * Initialises all styles for the given workbook.
         *
         * @param workbook the workbook for which to create styles
         */
        StyleHelper(XSSFWorkbook workbook) {
            this.titleStyle = createTitleStyle(workbook);
            this.sectionStyle = createSectionStyle(workbook);
            this.headerStyle = createHeaderStyle(workbook);
            this.cellStyle = createCellStyle(workbook);
            this.numberStyle = createNumberStyle(workbook);
        }
        
        /**
         * Bold, 12-point font; used for the run title at the top of each sheet.
         */
        private XSSFCellStyle createTitleStyle(XSSFWorkbook workbook) {
            XSSFFont font = workbook.createFont();
            font.setBold(true);
            font.setFontHeightInPoints((short) 12);
            XSSFCellStyle style = workbook.createCellStyle();
            style.setFont(font);
            return style;
        }
        
        /**
         * Bold white text on a dark-blue background; used for section headings ("EVASION ACCURACY" / "CLARITY
         * ACCURACY").
         */
        private XSSFCellStyle createSectionStyle(XSSFWorkbook workbook) {
            XSSFFont font = workbook.createFont();
            font.setBold(true);
            font.setColor(IndexedColors.WHITE.getIndex());
            XSSFCellStyle style = workbook.createCellStyle();
            style.setFont(font);
            style.setFillForegroundColor(IndexedColors.DARK_BLUE.getIndex());
            style.setFillPattern(FillPatternType.SOLID_FOREGROUND);
            return style;
        }
        
        /**
         * Bold text on a light-grey background with thin borders; used for table column headers.
         */
        private XSSFCellStyle createHeaderStyle(XSSFWorkbook workbook) {
            XSSFFont font = workbook.createFont();
            font.setBold(true);
            XSSFCellStyle style = workbook.createCellStyle();
            style.setFont(font);
            style.setFillForegroundColor(IndexedColors.GREY_25_PERCENT.getIndex());
            style.setFillPattern(FillPatternType.SOLID_FOREGROUND);
            applyBorders(style);
            return style;
        }
        
        /**
         * Plain text with thin borders; used for string data cells.
         */
        private XSSFCellStyle createCellStyle(XSSFWorkbook workbook) {
            XSSFCellStyle style = workbook.createCellStyle();
            applyBorders(style);
            return style;
        }
        
        /**
         * Right-aligned with thin borders; used for numeric data cells (counts and percentages).
         */
        private XSSFCellStyle createNumberStyle(XSSFWorkbook workbook) {
            XSSFCellStyle style = workbook.createCellStyle();
            style.setAlignment(HorizontalAlignment.RIGHT);
            applyBorders(style);
            return style;
        }
        
        /**
         * Applies a thin border on all four sides of the given cell style.
         *
         * @param style the style to modify in place
         */
        private void applyBorders(XSSFCellStyle style) {
            style.setBorderTop(BorderStyle.THIN);
            style.setBorderBottom(BorderStyle.THIN);
            style.setBorderLeft(BorderStyle.THIN);
            style.setBorderRight(BorderStyle.THIN);
        }
        
        /** @return the style used for the run-title row */
        XSSFCellStyle getTitleStyle() { return titleStyle; }
        
        /** @return the style used for section-header rows */
        XSSFCellStyle getSectionStyle() { return sectionStyle; }
        
        /** @return the style used for table column-header rows */
        XSSFCellStyle getHeaderStyle() { return headerStyle; }
        
        /** @return the style used for string data cells */
        XSSFCellStyle getCellStyle() { return cellStyle; }
        
        /** @return the style used for numeric data cells */
        XSSFCellStyle getNumberStyle() { return numberStyle; }
    }
}
