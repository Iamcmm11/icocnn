# LOCATA Model Compare

- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`
- available recordings: `task1=13, task3=5, task5=5, total=23`
- MAC: stage-3 `model_profile.mac_proxy_total`

## With Silences

| Model | Params | MAC | Task1 Best | Delta | Task1 Mean | Delta | Task3 Best | Delta | Task3 Mean | Delta | Task5 Best | Delta | Task5 Mean | Delta | Std | Delta | Median | Delta | Average | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 290017 | - | 2.8532 | +0.0000 | 6.2600 | +0.0000 | 6.6209 | +0.0000 | 10.7483 | +0.0000 | 8.4122 | +0.0000 | 12.4059 | +0.0000 | 4.1621 | +0.0000 | 7.1128 | +0.0000 | 8.5718 | +0.0000 |
| replace_1d_with_maba | 282529 | - | 1.4011 | -1.4521 | 5.2364 | -1.0236 | 6.8568 | +0.2359 | 10.7964 | +0.0481 | 8.1015 | -0.3106 | 11.0364 | -1.3694 | 4.1560 | -0.0062 | 6.8568 | -0.2560 | 7.7060 | -0.8658 |
| ablation_no_gate | 297769 | - | 2.5128 | -0.3404 | 5.6413 | -0.6187 | 6.1392 | -0.4816 | 11.1449 | +0.3966 | 7.4798 | -0.9323 | 11.9712 | -0.4347 | 4.7086 | +0.5465 | 6.9040 | -0.2088 | 8.2138 | -0.3580 |
| IFAN | 125457 | - | 1.9414 | -0.9118 | 5.0884 | -1.1715 | 6.6484 | +0.0275 | 9.8438 | -0.9045 | 6.8890 | -1.5232 | 11.0712 | -1.3346 | 4.0175 | -0.1447 | 6.6484 | -0.4644 | 7.4228 | -1.1489 |
| IFAN_Maba | 133297 | - | 1.6563 | -1.1969 | 4.9583 | -1.3017 | 5.9741 | -0.6467 | 9.9412 | -0.8072 | 6.9318 | -1.4803 | 11.7462 | -0.6597 | 4.5717 | +0.4096 | 6.9318 | -0.1810 | 7.5172 | -1.0546 |
| IFAN_80 | 125457 | 459532800 | 2.1520 | -0.7012 | **5.0969** | **-1.1631** | 6.2791 | -0.3418 | **9.0160** | **-1.7323** | 7.5993 | -0.8129 | **11.0393** | **-1.3666** | 3.8459 | -0.3163 | 6.4419 | -0.6709 | 7.2407 | -1.3310 |
| IFAN_C8_R2 | 31561 | 115211520 | 2.5956 | -0.2575 | **5.4455** | **-0.8144** | 5.5883 | -1.0326 | **10.1375** | **-0.6108** | 7.6773 | -0.7349 | **11.8516** | **-0.5543** | 4.1779 | +0.0157 | 6.5339 | -0.5789 | 7.8581 | -0.7136 |
| IFAN_C8_R3 (failed ref) | 31561 | 460846080 | 2.6951 | -0.1581 | 6.0880 | -0.1720 | 8.0931 | +1.4722 | 11.0955 | +0.3472 | 7.8560 | -0.5562 | 12.5763 | +0.1704 | 4.4462 | +0.2840 | 8.0931 | +0.9803 | 8.5871 | +0.0153 |
| IFAN_LC | 125457 | - | 1.9897 | -0.8635 | 5.2566 | -1.0034 | 7.4871 | +0.8662 | 10.0523 | -0.6960 | 6.0849 | -2.3273 | 12.5251 | +0.1193 | 5.0923 | +0.9302 | 7.4871 | +0.3743 | 7.8793 | -0.6925 |

## Without Silences

| Model | Params | MAC | Task1 Best | Delta | Task1 Mean | Delta | Task3 Best | Delta | Task3 Mean | Delta | Task5 Best | Delta | Task5 Mean | Delta | Std | Delta | Median | Delta | Average | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 290017 | - | 2.8647 | +0.0000 | 6.0413 | +0.0000 | 6.3474 | +0.0000 | 7.3898 | +0.0000 | 7.3488 | +0.0000 | 10.0118 | +0.0000 | 3.0529 | +0.0000 | 7.1740 | +0.0000 | 7.1976 | +0.0000 |
| replace_1d_with_maba | 282529 | - | 1.0473 | -1.8174 | 5.3023 | -0.7390 | 5.7494 | -0.5980 | 7.2460 | -0.1437 | 7.1990 | -0.1498 | 8.3433 | -1.6684 | 2.4706 | -0.5823 | 6.2652 | -0.9089 | 6.3859 | -0.8116 |
| ablation_no_gate | 297769 | - | 2.2594 | -0.6053 | 5.4918 | -0.5495 | 5.9248 | -0.4227 | 7.1744 | -0.2154 | 6.9931 | -0.3557 | 9.6929 | -0.3189 | 3.5287 | +0.4758 | 6.9061 | -0.2679 | 6.7709 | -0.4267 |
| IFAN | 125457 | - | 2.0958 | -0.7689 | 5.1980 | -0.8433 | 7.2208 | +0.8733 | 8.7194 | +1.3297 | 6.0021 | -1.3467 | 8.8576 | -1.1541 | 3.1258 | +0.0729 | 6.2717 | -0.9023 | 6.7591 | -0.4385 |
| IFAN_Maba | 133297 | - | 1.8093 | -1.0554 | 5.0586 | -0.9827 | 6.5756 | +0.2282 | 8.7502 | +1.3605 | 6.1915 | -1.1572 | 8.9852 | -1.0266 | 3.4925 | +0.4396 | 6.5756 | -0.5984 | 6.7147 | -0.4829 |
| IFAN_80 | 125457 | 459532800 | 2.1320 | -0.7328 | **5.1058** | **-0.9355** | 6.0259 | -0.3215 | **7.1407** | **-0.2490** | 6.4700 | -0.8788 | **8.4228** | **-1.5890** | 2.6491 | -0.4038 | 6.4700 | -0.7040 | 6.2693 | -0.9283 |
| IFAN_C8_R2 | 31561 | 115211520 | 2.5505 | -0.3143 | **5.4777** | **-0.5636** | 5.6990 | -0.6484 | **8.3455** | **+0.9557** | 6.8928 | -0.4560 | **9.9599** | **-0.0519** | 3.3424 | +0.2895 | 6.8928 | -0.2812 | 7.0755 | -0.1221 |
| IFAN_C8_R3 (failed ref) | 31561 | 460846080 | 2.7248 | -0.1400 | 5.8945 | -0.1468 | 7.6250 | +1.2776 | 9.0824 | +1.6927 | 6.5326 | -0.8162 | 10.4713 | +0.4595 | 3.8183 | +0.7654 | 6.8692 | -0.3048 | 7.5825 | +0.3849 |
| IFAN_LC | 125457 | - | 1.6380 | -1.2268 | 5.4251 | -0.6161 | 7.1086 | +0.7612 | 8.5196 | +1.1298 | 5.4433 | -1.9055 | 10.8190 | +0.8072 | 4.9584 | +1.9055 | 7.0689 | -0.1051 | 7.2704 | +0.0728 |

## Focused Comparison

Primary comparison should center on `baseline`, `IFAN_80`, and `IFAN_C8_R2`.

| Comparison | Params | Params Change | MAC | MAC Change | With Silences Average | Without Silences Average | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `IFAN_80` vs `baseline` | `290017 -> 125457` | `56.7%` reduction | `n/a -> 459532800` | `n/a` | `-1.3311 deg` (`15.5%` better) | `-0.9283 deg` (`12.9%` better) | Full IFAN-80 is the strongest accuracy-oriented reference among the three key models. |
| `IFAN_C8_R2` vs `baseline` | `290017 -> 31561` | `89.1%` reduction | `n/a -> 115211520` | `n/a` | `-0.7137 deg` (`8.3%` better) | `-0.1221 deg` (`1.7%` better) | Even after aggressive compression, `C8_R2` still improves over the baseline on average LOCATA RMSAE. |
| `IFAN_C8_R2` vs `IFAN_80` | `125457 -> 31561` | `74.8%` reduction | `459532800 -> 115211520` | `74.9%` reduction | `+0.6174 deg` (`8.5%` loss) | `+0.8062 deg` (`12.9%` loss) | This is the key edge trade-off: modest average accuracy loss in exchange for a major compute reduction. |

- `IFAN_C8_R2` is the main edge result: it remains better than `baseline`, while being much cheaper than `IFAN_80`.
- `IFAN_C8_R3` is kept only as a failure reference: increasing `r` did not recover the `C8_R2` accuracy gap, and it almost removed the MAC advantage.

## Edge Trade-off vs IFAN_80

Using `IFAN_80` as the accuracy-oriented reference:

| Model | Params Reduction | MAC Reduction | With Silences Avg Delta (deg) | With Silences Avg Delta (%) | Without Silences Avg Delta (deg) | Without Silences Avg Delta (%) |
| --- | --- | --- | --- | --- | --- | --- |
| IFAN_C8_R2 | 74.8% | 74.9% | +0.6174 | +8.5% | +0.8062 | +12.9% |
| IFAN_C8_R3 (failed ref) | 74.8% | -0.3% | +1.3464 | +18.6% | +1.3132 | +20.9% |

- `IFAN_C8_R2` is the more meaningful edge point: it cuts about `74.8%` parameters and `74.9%` MAC while increasing LOCATA average RMSAE by only about `0.62 deg` to `0.81 deg`.
- `IFAN_C8_R3` keeps the same parameter reduction as `C8_R2`, but its MAC is essentially unchanged relative to `IFAN_80`, while the average RMSAE regression is noticeably larger.
