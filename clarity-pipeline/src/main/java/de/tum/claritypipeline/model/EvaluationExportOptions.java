package de.tum.claritypipeline.model;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class EvaluationExportOptions {
    @Builder.Default
    private int roundToDigits = 3;

    @Builder.Default
    private short headerFontSize = 12;

    @Builder.Default
    private String numberFormat = "0.00";

    @Builder.Default
    private String sheetName = "Evaluation Export";

}
