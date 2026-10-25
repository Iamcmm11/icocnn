# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/stage3/ifan_stage3_long80_c32_no_prefusion_pool_20260621_152848/checkpoints/epoch_050.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `7.8289 deg`
- IFAN without silences mean RMSAE: `7.0954 deg`
- Baseline with silences mean RMSAE: `7.7387 deg`
- Baseline without silences mean RMSAE: `7.1201 deg`
- Delta vs baseline with silences: `+0.0903 deg`
- Delta vs baseline without silences: `-0.0247 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `5.9137 deg`
- IFAN without silences mean RMSAE: `6.0150 deg`
- Baseline with silences mean RMSAE: `5.9252 deg`
- Baseline without silences mean RMSAE: `6.1232 deg`
- Delta vs baseline with silences: `-0.0114 deg`
- Delta vs baseline without silences: `-0.1082 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `10.1907 deg`
- IFAN without silences mean RMSAE: `8.7292 deg`
- Baseline with silences mean RMSAE: `9.0915 deg`
- Baseline without silences mean RMSAE: `7.9956 deg`
- Delta vs baseline with silences: `+1.0992 deg`
- Delta vs baseline without silences: `+0.7337 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `10.4468 deg`
- IFAN without silences mean RMSAE: `8.2706 deg`
- Baseline with silences mean RMSAE: `11.1009 deg`
- Baseline without silences mean RMSAE: `8.8365 deg`
- Delta vs baseline with silences: `-0.6541 deg`
- Delta vs baseline without silences: `-0.5659 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
