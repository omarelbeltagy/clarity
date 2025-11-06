package de.tum.claritypipeline.model.config;

import lombok.*;

/**
 * Configuration options when exporting evaluation results to an Excel file.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class EvaluationExportProperties {
    /**
     * Number of decimal places to round numerical values to in the export.
     */
    @Builder.Default
    private int roundToDigits = 3;

    /**
     * Font size for headers in the exported Excel sheet.
     */
    @Builder.Default
    private short headerFontSize = 12;

    /**
     * Format string for numerical values in the export.
     */
    @Builder.Default
    private String numberFormat = "0.00";

    /**
     * Name of the Excel sheet where evaluation results will be exported.
     */
    @Builder.Default
    private String sheetName = "Evaluation Export";

}
