# LOCATA Model Compare

- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`
- available recordings: `task1=13, task3=5, task5=5, total=23`
- MAC: stage-3 `model_profile.mac_proxy_total`

## Simulated Four-Scene Compare

- source: each run's `baseline_compare.json`
- metric: RMSAE deg; Delta is vs the original baseline checkpoint
- scenarios: `scene_1=30dB/T60=0.2s`, `scene_2=30dB/T60=0.8s`, `scene_3=5dB/T60=0.8s`, `scene_4=5dB/T60=1.4s`
- hard mean: average of `scene_3` and `scene_4`

| Model | Params | MAC | Scene1 | Delta | Scene2 | Delta | Scene3 | Delta | Scene4 | Delta | Mean | Delta | Hard Mean | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IFAN_C8_R2 | 31561 | 115211520 | 5.2905 | -0.0893 | 6.0883 | +0.5069 | 10.5562 | +1.3803 | 17.9796 | +4.2401 | 9.9787 | +1.5095 | 14.2679 | +2.8102 |
| IFAN_C8_R2_map_maba | 33409 | 115221840 | 5.7687 | +0.3890 | 6.1328 | +0.5514 | 9.7947 | +0.6188 | 17.9463 | +4.2068 | 9.9106 | +1.4415 | 13.8705 | +2.4128 |
| ifan_c8_r2_maba_pre_readout_best | 32353 | 116213760 | 5.1647 | -0.2150 | 5.6017 | +0.0203 | 9.4865 | +0.3106 | 13.9095 | +0.1700 | 8.5406 | +0.0715 | 11.6980 | +0.2403 |
| ifan_c8_r2_maba_pre_readout_best_retrain_current_arch_20260526 | 32321 | 116213760 | 4.3319 | -1.0478 | 5.4872 | -0.0942 | 11.5318 | +2.3559 | 16.1447 | +2.4052 | 9.3739 | +0.9048 | 13.8383 | +2.3806 |
| ifan_c8_r2_maba_dual_refine_best | 33201 | 116218344 | 5.1004 | -0.2793 | 5.5467 | -0.0347 | 9.4702 | +0.2942 | 14.6532 | +0.9137 | 8.6926 | +0.2235 | 12.0617 | +0.6040 |

### Why `pre_readout` helps scene_4

`ifan_c8_r2_maba_pre_readout_best` adds a `FeatureMABATemporalRefiner` at `map_refiner_position = "pre_readout"`. In code, this means the refiner runs after `final_block` and before `channel_readout`: it sees tensors shaped like `(B, T, C, R, charts, H, W)`, so it can still refine the 8 feature channels and all 6 icosahedral regions before they are collapsed into a single score map. By contrast, `IFAN_C8_R2_map_maba` uses `MABATemporalRefiner` after `channel_readout`, after `squeeze` and `max(dim=2)`, so it only sees the final scalar map `(B, T, charts, H, W)` after channel and region information has already been discarded.

The actual MABA block is a causal temporal refiner: channel/map projection, depthwise temporal convolution, layer norm/dropout, and a gated state update when `use_state = true`, followed by a residual delta. For scene_4 (`5dB/T60=1.4s`), the input is both noisy and highly reverberant, so the final scalar map is more likely to contain unstable peaks. Refining before readout preserves weak multi-channel and multi-region evidence long enough for the temporal state to suppress frame-level spikes before the hard region max and SoftArgMax. This matches the result: scene_4 RMSAE drops from `17.9796` (`IFAN_C8_R2`) and `17.9463` (`map_maba`) to `13.9095`, almost back to the baseline (`+0.1700 deg` delta).

`dual_refine` keeps the same strong pre-readout refiner but adds a second weak pre-SoftArgMax map refiner. Its scene_1/2/3 numbers are slightly better than `pre_readout_best`, but scene_4 worsens to `14.6532`. The code path suggests the extra scalar-map refiner can re-shape already-collapsed heatmaps; in the hardest reverberant case that likely over-smooths or shifts the peak after the useful feature-level correction has already happened. So the practical conclusion is that `pre_readout_best` works mainly because it refines temporal evidence before the lossy `channel_readout -> region max -> SoftArgMax` steps, while the map-level refiners operate too late for scene_4.

### 20260526 retrain note

`ifan_c8_r2_maba_pre_readout_best_retrain_current_arch_20260526` is the new run using the current residual/feature-enhancement logic. It is appended below the original `ifan_c8_r2_maba_pre_readout_best` row rather than replacing it. This retrain improves scene_1/2 but regresses scene_3/4 in the simulated four-scene suite, and LOCATA is nearly tied with the baseline overall while improving task1 and regressing task5.

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
| IFAN_C8_R2_map_maba | 33409 | 115221840 | 2.5737 | -0.2795 | 4.8651 | -1.3949 | 5.9471 | -0.6738 | 9.9524 | -0.7959 | 7.9195 | -0.4927 | 13.0768 | +0.6709 | 5.0212 | +0.8591 | 6.8274 | -0.2854 | 7.7562 | -0.8156 |
| ifan_c8_r2_maba_pre_readout_best | 32353 | 116213760 | 1.9856 | -0.8676 | 5.5415 | -0.7185 | 6.3290 | -0.2919 | 10.2378 | -0.5105 | 7.6434 | -0.7688 | 11.2159 | -1.1900 | 3.7426 | -0.4195 | 6.5392 | -0.5736 | 7.7960 | -0.7758 |
| ifan_c8_r2_maba_pre_readout_best_retrain_current_arch_20260526 | 32321 | 116213760 | 2.4858 | -0.8472 | 5.3317 | -0.5935 | 5.4690 | -0.0825 | 10.0334 | +0.9419 | 7.6402 | +0.3873 | 12.2045 | +1.1036 | 4.4835 | +0.8447 | 7.4794 | +0.6740 | 7.8479 | +0.1092 |
| ifan_c8_r2_maba_dual_refine_best | 33201 | 116218344 | 2.6629 | -0.1903 | 5.2623 | -0.9977 | 5.6569 | -0.9640 | 10.2160 | -0.5323 | 7.5583 | -0.8539 | 11.5660 | -0.8399 | 4.1180 | -0.0441 | 6.1003 | -1.0125 | 7.7096 | -0.8622 |
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
| IFAN_C8_R2_map_maba | 33409 | 115221840 | 2.4981 | -0.3666 | 4.7522 | -1.2891 | 5.5641 | -0.7833 | 8.0200 | +0.6302 | 7.0664 | -0.2824 | 11.1730 | +1.1612 | 4.5380 | +1.4851 | 6.8303 | -0.3437 | 6.8584 | -0.3392 |
| ifan_c8_r2_maba_pre_readout_best | 32353 | 116213760 | 2.2507 | -0.6140 | 5.6588 | -0.3825 | 6.4300 | +0.0826 | 8.5392 | +1.1494 | 6.4051 | -0.9437 | 8.5478 | -1.4640 | 2.5667 | -0.4862 | 6.9061 | -0.2679 | 6.9130 | -0.2846 |
| ifan_c8_r2_maba_pre_readout_best_retrain_current_arch_20260526 | 32321 | 116213760 | 2.4842 | -0.8610 | 5.3039 | -0.8193 | 5.2909 | -0.2012 | 8.4705 | +0.4749 | 6.4787 | -0.3423 | 10.4991 | +1.6626 | 4.0575 | +1.4756 | 6.8941 | +0.1085 | 7.1217 | +0.0016 |
| ifan_c8_r2_maba_dual_refine_best | 33201 | 116218344 | 2.4380 | -0.4267 | 5.3078 | -0.7335 | 5.4420 | -0.9054 | 8.3030 | +0.9132 | 6.6369 | -0.7119 | 9.1205 | -0.8913 | 2.8180 | -0.2349 | 6.3248 | -0.8492 | 6.7878 | -0.4098 |
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
