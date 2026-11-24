# Splits and folds (cell level)

config `d76c17023045a305`
seeds: split 20260101, fold 20260102
test 6 cells (one per region), calibration 6, development 36

## Cells and rows by role and region

| role | region | cells | rows | >25 | >50 | nearest dev cell (km) |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 1 | 1 | 3,416 | 23 | 0 | 5.0 |
| calibration | 2 | 1 | 5,058 | 52 | 4 | 11.2 |
| calibration | 3 | 1 | 2,861 | 9 | 0 | 5.0 |
| calibration | 4 | 1 | 4,992 | 31 | 3 | 58.3 |
| calibration | 5 | 1 | 4,668 | 60 | 4 | 5.0 |
| calibration | 6 | 1 | 5,027 | 16 | 5 | 256.7 |
| development | 1 | 6 | 16,850 | 188 | 10 |  |
| development | 2 | 7 | 31,526 | 209 | 19 |  |
| development | 3 | 6 | 26,174 | 113 | 25 |  |
| development | 4 | 8 | 27,085 | 189 | 20 |  |
| development | 5 | 5 | 24,797 | 172 | 11 |  |
| development | 6 | 4 | 18,883 | 74 | 16 |  |
| spatial_test | 1 | 1 | 5,073 | 51 | 3 | 5.0 |
| spatial_test | 2 | 1 | 5,060 | 30 | 3 | 30.4 |
| spatial_test | 3 | 1 | 5,027 | 32 | 5 | 68.0 |
| spatial_test | 4 | 1 | 5,013 | 23 | 4 | 62.6 |
| spatial_test | 5 | 1 | 5,007 | 45 | 2 | 21.2 |
| spatial_test | 6 | 1 | 1,095 | 1 | 0 | 5.0 |

## Picked cells

| cell | role | region | rows 2010-2023 | nearest dev cell (km) |
|---|---|---|---:|---:|
| 918737 | calibration | 1 | 3,416 | 5.0 |
| 928317 | calibration | 2 | 5,058 | 11.2 |
| 864472 | calibration | 3 | 2,861 | 5.0 |
| 933610 | calibration | 4 | 4,992 | 58.3 |
| 930445 | calibration | 5 | 4,668 | 5.0 |
| 863269 | calibration | 6 | 5,027 | 256.7 |
| 918734 | spatial_test | 1 | 5,073 | 5.0 |
| 910214 | spatial_test | 2 | 5,060 | 30.4 |
| 895350 | spatial_test | 3 | 5,027 | 68.0 |
| 958101 | spatial_test | 4 | 5,013 | 62.6 |
| 924048 | spatial_test | 5 | 5,007 | 21.2 |
| 856931 | spatial_test | 6 | 1,095 | 5.0 |

## Fold sizes

| fold | cells | rows | regions |
|---:|---:|---:|---|
| 0 | 4 | 16,030 | [3, 4] |
| 1 | 4 | 11,398 | [1, 4, 5] |
| 2 | 4 | 18,962 | [2, 6] |
| 3 | 4 | 13,673 | [1, 2] |
| 4 | 4 | 17,129 | [2, 3, 4, 6] |
| 5 | 4 | 20,063 | [2, 3, 5] |
| 6 | 3 | 11,783 | [2, 3, 5] |
| 7 | 3 | 8,757 | [3, 4, 5] |
| 8 | 3 | 14,896 | [3, 4, 5] |
| 9 | 3 | 12,624 | [1, 2] |
