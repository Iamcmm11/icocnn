# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/stage3/ifan_stage3_long80_c32_no_prefusion_pool_20260621_152848/checkpoints/epoch_020.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `8.1975 deg`
- IFAN without silences mean RMSAE: `7.3386 deg`
- Baseline with silences mean RMSAE: `7.7387 deg`
- Baseline without silences mean RMSAE: `7.1201 deg`
- Delta vs baseline with silences: `+0.4588 deg`
- Delta vs baseline without silences: `+0.2185 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `6.0181 deg`
- IFAN without silences mean RMSAE: `6.0834 deg`
- Baseline with silences mean RMSAE: `5.9252 deg`
- Baseline without silences mean RMSAE: `6.1232 deg`
- Delta vs baseline with silences: `+0.0929 deg`
- Delta vs baseline without silences: `-0.0398 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `10.3812 deg`
- IFAN without silences mean RMSAE: `8.8783 deg`
- Baseline with silences mean RMSAE: `9.0915 deg`
- Baseline without silences mean RMSAE: `7.9956 deg`
- Delta vs baseline with silences: `+1.2897 deg`
- Delta vs baseline without silences: `+0.8827 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `11.6803 deg`
- IFAN without silences mean RMSAE: `9.0623 deg`
- Baseline with silences mean RMSAE: `11.1009 deg`
- Baseline without silences mean RMSAE: `8.8365 deg`
- Delta vs baseline with silences: `+0.5793 deg`
- Delta vs baseline without silences: `+0.2258 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
