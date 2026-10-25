# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/stage3/ifan_stage3_long80_c32_no_prefusion_pool_20260621_152848/checkpoints/epoch_010.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `9.0535 deg`
- IFAN without silences mean RMSAE: `8.1087 deg`
- Baseline with silences mean RMSAE: `7.7387 deg`
- Baseline without silences mean RMSAE: `7.1201 deg`
- Delta vs baseline with silences: `+1.3149 deg`
- Delta vs baseline without silences: `+0.9886 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `7.0348 deg`
- IFAN without silences mean RMSAE: `7.1005 deg`
- Baseline with silences mean RMSAE: `5.9252 deg`
- Baseline without silences mean RMSAE: `6.1232 deg`
- Delta vs baseline with silences: `+1.1097 deg`
- Delta vs baseline without silences: `+0.9773 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `10.7940 deg`
- IFAN without silences mean RMSAE: `9.1713 deg`
- Baseline with silences mean RMSAE: `9.0915 deg`
- Baseline without silences mean RMSAE: `7.9956 deg`
- Delta vs baseline with silences: `+1.7025 deg`
- Delta vs baseline without silences: `+1.1757 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `12.5618 deg`
- IFAN without silences mean RMSAE: `9.6674 deg`
- Baseline with silences mean RMSAE: `11.1009 deg`
- Baseline without silences mean RMSAE: `8.8365 deg`
- Delta vs baseline with silences: `+1.4608 deg`
- Delta vs baseline without silences: `+0.8309 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
