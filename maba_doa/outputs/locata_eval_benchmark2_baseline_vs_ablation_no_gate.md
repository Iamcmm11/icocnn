# MABA-DOA LOCATA Baseline vs MABA

- subset: `eval`
- array: `benchmark2`
- tasks: `1, 3, 5`
- baseline checkpoint: `/home/cmm/icocnn/maba_doa/outputs/maba_doa_r2_baseline_20260406_220454/model.bin`
- maba checkpoint: `/home/cmm/icocnn/maba_doa/outputs/maba_doa_r2_ablation_no_gate_retry25_20260407_125521/model.bin`
- LOCATA is an evaluation dataset, not a training dataset.

## Overall Summary

| Metric | Baseline | MABA | Delta (MABA-Baseline) | Better |
| --- | ---: | ---: | ---: | --- |
| With silences mean RMSAE (deg) | 8.5718 | 8.2138 | -0.3580 | maba |
| Without silences mean RMSAE (deg) | 7.1976 | 6.7709 | -0.4267 | maba |

## Task Summary

| Task | Count | Baseline WS mean | MABA WS mean | Delta WS | Baseline NS mean | MABA NS mean | Delta NS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task1 | 13 | 6.2600 | 5.6413 | -0.6187 | 6.0413 | 5.4918 | -0.5495 |
| task3 | 5 | 10.7483 | 11.1449 | 0.3966 | 7.3898 | 7.1744 | -0.2154 |
| task5 | 5 | 12.4059 | 11.9712 | -0.4347 | 10.0118 | 9.6929 | -0.3189 |

## Recording-level Results

| Task | Recording | Baseline WS | MABA WS | Delta WS | Baseline NS | MABA NS | Delta NS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | recording1 | 9.0574 | 7.8600 | -1.1974 | 8.5669 | 7.9819 | -0.5850 |
| 1 | recording10 | 6.6491 | 4.3245 | -2.3246 | 6.4128 | 3.7463 | -2.6665 |
| 1 | recording11 | 2.8532 | 3.9765 | 1.1233 | 3.2917 | 4.1662 | 0.8745 |
| 1 | recording12 | 5.4916 | 4.2088 | -1.2828 | 5.4411 | 4.1377 | -1.3034 |
| 1 | recording13 | 5.5359 | 5.7908 | 0.2549 | 4.5041 | 5.0606 | 0.5566 |
| 1 | recording2 | 2.8580 | 3.2959 | 0.4378 | 2.8647 | 3.3020 | 0.4372 |
| 1 | recording3 | 9.8820 | 10.1899 | 0.3080 | 9.8814 | 10.2443 | 0.3629 |
| 1 | recording4 | 7.1042 | 6.7655 | -0.3386 | 7.1740 | 6.9061 | -0.2679 |
| 1 | recording5 | 5.1094 | 6.6408 | 1.5314 | 5.1748 | 6.4441 | 1.2693 |
| 1 | recording6 | 5.5970 | 2.6028 | -2.9942 | 4.4998 | 2.2594 | -2.2404 |
| 1 | recording7 | 3.4908 | 2.5128 | -0.9780 | 3.8801 | 2.6699 | -1.2102 |
| 1 | recording8 | 6.8891 | 3.4237 | -3.4653 | 6.3554 | 2.7716 | -3.5837 |
| 1 | recording9 | 10.8621 | 11.7450 | 0.8829 | 10.4899 | 11.7033 | 1.2135 |
| 3 | recording1 | 10.3574 | 9.7104 | -0.6469 | 7.5817 | 6.5540 | -1.0277 |
| 3 | recording2 | 16.0322 | 16.2022 | 0.1700 | 8.8256 | 9.2065 | 0.3809 |
| 3 | recording3 | 13.6184 | 16.7687 | 3.1503 | 7.0157 | 7.2474 | 0.2316 |
| 3 | recording4 | 6.6209 | 6.1392 | -0.4816 | 6.3474 | 5.9248 | -0.4227 |
| 3 | recording5 | 7.1128 | 6.9040 | -0.2088 | 7.1783 | 6.9392 | -0.2391 |
| 5 | recording1 | 20.2156 | 20.6525 | 0.4369 | 18.0850 | 18.9574 | 0.8724 |
| 5 | recording2 | 10.7423 | 9.1817 | -1.5606 | 8.9437 | 7.1110 | -1.8327 |
| 5 | recording3 | 13.0728 | 12.4137 | -0.6590 | 8.2811 | 7.5406 | -0.7406 |
| 5 | recording4 | 9.5866 | 10.1281 | 0.5415 | 7.4002 | 7.8625 | 0.4623 |
| 5 | recording5 | 8.4122 | 7.4798 | -0.9323 | 7.3488 | 6.9931 | -0.3557 |

## Short Conclusion

- With silences better model: `maba`
- Without silences better model: `maba`
- task1: with silences=`maba`, without silences=`maba`
- task3: with silences=`baseline`, without silences=`maba`
- task5: with silences=`maba`, without silences=`maba`
