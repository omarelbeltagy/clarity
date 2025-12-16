package de.tum.claritypipeline.model.config;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import de.tum.clarityutils.JacksonUtils;
import lombok.*;

import java.io.File;
import java.io.IOException;

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
    @JsonProperty("round_to_digits")
    @JsonPropertyDescription("Number of decimal places used when rounding metrics in the Excel sheet.")
    private int roundToDigits = 3;

    /**
     * Font size for headers in the exported Excel sheet.
     */
    @Builder.Default
    @JsonProperty("header_font_size")
    @JsonPropertyDescription("Font size (pt) applied to header cells in the exported workbook.")
    private short headerFontSize = 12;

    /**
     * Format string for numerical values in the export.
     */
    @Builder.Default
    @JsonProperty("number_format")
    @JsonPropertyDescription("Excel number-format string applied to metric cells (e.g., 0.0000).")
    private String numberFormat = "0.00";

    /**
     * Name of the Excel sheet where evaluation results will be exported.
     */
    @Builder.Default
    @JsonProperty("sheet_name")
    @JsonPropertyDescription("Worksheet name inside the generated XLSX file.")
    private String sheetName = "Evaluation Export";

    public static EvaluationExportProperties load(String path) throws IOException {
        if (path == null || path.isEmpty()) {
            throw new IOException("No path specified for EvaluationExportProperties file.");
        }
        ObjectMapper mapper = JsonMapper.builder(new YAMLFactory())
                                        .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_ENUMS, true)
                                        .configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, false)
                                        .build();
        return JacksonUtils.readAndInit(mapper, new File(path), EvaluationExportProperties.class);
    }

}
