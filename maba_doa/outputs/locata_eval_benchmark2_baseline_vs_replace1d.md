# MABA-DOA LOCATA Baseline vs MABA

- subset: `eval`
- array: `benchmark2`
- tasks: `1, 3, 5`
- baseline checkpoint: `/home/cmm/icocnn/maba_doa/outputs/maba_doa_r2_baseline_20260406_220454/model.bin`
- maba checkpoint: `/home/cmm/icocnn/maba_doa/outputs/maba_doa_r2_replace_1d_with_maba_25ep_20260419_175206/model.bin`
- LOCATA is an evaluation dataset, not a training dataset.

## Overall Summary

| Metric | Baseline | MABA | Delta (MABA-Baseline) | Better |
| --- | ---: | ---: | ---: | --- |
| With silences mean RMSAE (deg) | 8.5718 | 7.7060 | -0.8658 | maba |
| Without silences mean RMSAE (deg) | 7.1976 | 6.3859 | -0.8116 | maba |

## Task Summary

| Task | Count | Baseline WS mean | MABA WS mean | Delta WS | Baseline NS mean | MABA NS mean | Delta NS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task1 | 13 | 6.2600 | 5.2364 | -1.0236 | 6.0413 | 5.3023 | -0.7390 |
| task3 | 5 | 10.7483 | 10.7964 | 0.0481 | 7.3898 | 7.2460 | -0.1437 |
| task5 | 5 | 12.4059 | 11.0364 | -1.3694 | 10.0118 | 8.3433 | -1.6684 |

## Recording-level Results

| Task | Recording | Baseline WS | MABA WS | Delta WS | Baseline NS | MABA NS | Delta NS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | recording1 | 9.0574 | 6.6593 | -2.3981 | 8.5669 | 6.2652 | -2.3017 |
| 1 | recording10 | 6.6491 | 7.1083 | 0.4592 | 6.4128 | 7.2033 | 0.7905 |
| 1 | recording11 | 2.8532 | 1.4011 | -1.4521 | 3.2917 | 1.0473 | -2.2444 |
| 1 | recording12 | 5.4916 | 5.2620 | -0.2296 | 5.4411 | 5.0860 | -0.3551 |
| 1 | recording13 | 5.5359 | 4.0921 | -1.4438 | 4.5041 | 4.5455 | 0.0415 |
| 1 | recording2 | 2.8580 | 2.7662 | -0.0918 | 2.8647 | 2.7876 | -0.0771 |
| 1 | recording3 | 9.8820 | 11.5691 | 1.6872 | 9.8814 | 11.5482 | 1.6668 |
| 1 | recording4 | 7.1042 | 4.8250 | -2.2792 | 7.1740 | 5.0808 | -2.0933 |
| 1 | recording5 | 5.1094 | 5.0848 | -0.0246 | 5.1748 | 4.9505 | -0.2243 |
| 1 | recording6 | 5.5970 | 5.9013 | 0.3043 | 4.4998 | 6.9715 | 2.4717 |
| 1 | recording7 | 3.4908 | 2.3405 | -1.1503 | 3.8801 | 2.4251 | -1.4550 |
| 1 | recording8 | 6.8891 | 4.8956 | -1.9935 | 6.3554 | 4.7983 | -1.5571 |
| 1 | recording9 | 10.8621 | 6.1675 | -4.6947 | 10.4899 | 6.2207 | -4.2692 |
| 3 | recording1 | 10.3574 | 7.8214 | -2.5359 | 7.5817 | 5.7494 | -1.8323 |
| 3 | recording2 | 16.0322 | 16.7029 | 0.6707 | 8.8256 | 10.0753 | 1.2497 |
| 3 | recording3 | 13.6184 | 15.5918 | 1.9734 | 7.0157 | 6.9157 | -0.1000 |
| 3 | recording4 | 6.6209 | 6.8568 | 0.2359 | 6.3474 | 6.2346 | -0.1128 |
| 3 | recording5 | 7.1128 | 7.0093 | -0.1035 | 7.1783 | 7.2550 | 0.0767 |
| 5 | recording1 | 20.2156 | 15.9931 | -4.2225 | 18.0850 | 11.1077 | -6.9773 |
| 5 | recording2 | 10.7423 | 9.4229 | -1.3194 | 8.9437 | 8.0359 | -0.9077 |
| 5 | recording3 | 13.0728 | 12.3387 | -0.7341 | 8.2811 | 7.5280 | -0.7532 |
| 5 | recording4 | 9.5866 | 9.3260 | -0.2607 | 7.4002 | 7.8462 | 0.4459 |
| 5 | recording5 | 8.4122 | 8.1015 | -0.3106 | 7.3488 | 7.1990 | -0.1498 |

## Short Conclusion

- With silences better model: `maba`
- Without silences better model: `maba`
- task1: with silences=`maba`, without silences=`maba`
- task3: with silences=`baseline`, without silences=`maba`
- task5: with silences=`maba`, without silences=`maba`
