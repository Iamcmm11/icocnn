# IFAN C32 No PoolIco LOCATA Epoch Sweep

- run: `IFAN_Edge/outputs/stage3/ifan_stage3_long80_c32_no_prefusion_pool_20260621_152848`
- checkpoint under question: `checkpoints/best_rmsae.pt`
- best simulated-validation epoch: `68`
- LOCATA subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Conclusion

`best_rmsae.pt` is also the best checkpoint among the saved LOCATA sweep points. The weaker result is therefore not caused by accidentally evaluating `last.pt`. The likely issue is not simple late-epoch overfitting after the validation optimum; it is a generalization/architecture trade-off of the high-capacity `C=32` no-`PoolIco` variant.

## LOCATA Sweep

| Checkpoint | With Silences Avg | Without Silences Avg | Task1 With | Task3 With | Task5 With | Task1 Without | Task3 Without | Task5 Without |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch_010 | 9.0535 | 8.1087 | 7.0348 | 10.7940 | 12.5618 | 7.1005 | 9.1713 | 9.6674 |
| epoch_020 | 8.1975 | 7.3386 | 6.0181 | 10.3812 | 11.6803 | 6.0834 | 8.8783 | 9.0623 |
| epoch_030 | 7.9004 | 7.1277 | 5.9417 | 10.1771 | 10.7163 | 6.0369 | 8.7514 | 8.3402 |
| epoch_040 | 7.8615 | 7.1099 | 5.9235 | 10.1158 | 10.6462 | 6.0174 | 8.6496 | 8.4107 |
| epoch_050 | 7.8289 | 7.0954 | 5.9137 | 10.1907 | 10.4468 | 6.0150 | 8.7292 | 8.2706 |
| epoch_060 | 7.7498 | 7.0634 | 5.8974 | 10.0805 | 10.2352 | 5.9762 | 8.7028 | 8.2507 |
| epoch_070 | 7.5808 | 6.9009 | 5.7209 | 9.7666 | 10.2307 | 5.8315 | 8.3236 | 8.2587 |
| epoch_080 | 7.5873 | 6.9024 | 5.7650 | 9.7526 | 10.1599 | 5.8850 | 8.2884 | 8.1618 |
| best_rmsae (epoch 68) | 7.5719 | 6.8309 | 5.6306 | 9.6484 | 10.5428 | 5.7557 | 8.0892 | 8.3680 |

## Training Curve Snapshot

| Epoch | Phase | Train Loss | Sim Val Loss | Sim Val RMSAE |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 1 | 0.020766 | 0.018538 | 6.9226 |
| 20 | 1 | 0.017665 | 0.017518 | 6.2586 |
| 30 | 2 | 0.019574 | 0.017071 | 5.9152 |
| 40 | 2 | 0.018832 | 0.016986 | 5.8687 |
| 50 | 2 | 0.017925 | 0.016910 | 5.8600 |
| 60 | 2 | 0.019056 | 0.016911 | 5.8784 |
| 68 | 2 | 0.018236 | 0.016822 | 5.7457 |
| 70 | 2 | 0.018277 | 0.016837 | 5.7732 |
| 80 | 2 | 0.017386 | 0.016824 | 5.7900 |
