# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/stage3/ifan_stage3_c16_saf_lite_4of8_phase1_20260518_014757/checkpoints/best_rmsae.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `8.1609 deg`
- IFAN without silences mean RMSAE: `6.8099 deg`
- Baseline with silences mean RMSAE: `7.7387 deg`
- Baseline without silences mean RMSAE: `7.1201 deg`
- Delta vs baseline with silences: `+0.4222 deg`
- Delta vs baseline without silences: `-0.3101 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `5.7703 deg`
- IFAN without silences mean RMSAE: `5.5109 deg`
- Baseline with silences mean RMSAE: `5.9252 deg`
- Baseline without silences mean RMSAE: `6.1232 deg`
- Delta vs baseline with silences: `-0.1548 deg`
- Delta vs baseline without silences: `-0.6122 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `9.9389 deg`
- IFAN without silences mean RMSAE: `7.5692 deg`
- Baseline with silences mean RMSAE: `9.0915 deg`
- Baseline without silences mean RMSAE: `7.9956 deg`
- Delta vs baseline with silences: `+0.8474 deg`
- Delta vs baseline without silences: `-0.4264 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `12.5983 deg`
- IFAN without silences mean RMSAE: `9.4280 deg`
- Baseline with silences mean RMSAE: `11.1009 deg`
- Baseline without silences mean RMSAE: `8.8365 deg`
- Delta vs baseline with silences: `+1.4974 deg`
- Delta vs baseline without silences: `+0.5915 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
