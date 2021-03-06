# Detailed experiments

### Selffer Network (BnB)

| model    |   B1B  |   B2B  |   B3B  |   B4B  |   B5B  |   B6B  |   B7B  |   B8B  |   B9B  |  B10B  |  B11B  |  B12B  |  B13B  |
|----------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| val_acc  | 0.9101 | 0.9226 | 0.9209 | 0.9213 | 0.9211 | 0.9213 | 0.9206 | 0.9209 | 0.9105 | 0.9129 | 0.9165 | 0.9166 | 0.9145 |
| test_acc | 0.9138 | 0.9231 | 0.9207 | 0.9199 | 0.9158 | 0.9211 | 0.9248 | 0.9260 | 0.9062 | 0.9106 | 0.9138 | 0.9133 | 0.9150 |

### Transfer Network (AnB)

| model | A1B | A2B | A3B | A4B | A5B | A6B | A7B | A8B | A9B | A10B | A11B | A12B | A13B |
|----------------------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| val_acc | 0.7497 | 0.6583 | 0.7359 | 0.5601 | 0.7383 | 0.7221 | 0.7477 | 0.7375 | 0.7618 | 0.7769 | 0.7073 | 0.7445 | 0.5934 |
| test_acc | 0.7126 | 0.6987 | 0.7704 | 0.6830 | 0.7212 | 0.7816 | 0.7080 | 0.7398 | 0.7650 | 0.7709 | 0.6189 | 0.7241 | 0.4676 |
| mean_time/epoch | 471s | 386s | 311s | 288s | 264s | 249s | 250s | 233s | 205s | 188s | 182s | 177s | 149s |
| early_stopping?(ep.) | yes(6) | yes(4) | yes(4) | yes(4) | yes(7) | yes(6) | yes(6) | yes(5) | yes(4) | yes(7) | yes(5) | yes(4) | yes(6) |

### Transfer Network (AnB+)

| model    |  A1B+  |  A2B+  |  A3B+  |  A4B+  |  A5B+  |  A6B+  |  A7B+  |  A8B+  |  A9B+  |  A10B+ |  A11B+ |  A12B+ |  A13B+ |
|----------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| val_acc  | 0.9218 | 0.9289 | 0.9269 | 0.9248 | 0.9109 | 0.9250 | 0.9246 | 0.9307 | 0.9238 | 0.9332 | 0.9279 | 0.9223 | 0.9339 |
| test_acc | 0.9087 | 0.9216 | 0.9216 | 0.9219 | 0.9023 | 0.9270 | 0.9260 | 0.9253 | 0.9243 | 0.9312 | 0.9312 | 0.9194 | 0.9309 |
