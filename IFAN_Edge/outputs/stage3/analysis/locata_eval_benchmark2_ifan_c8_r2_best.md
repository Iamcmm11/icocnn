# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_paper_original_20260505_222115/checkpoints/best_rmsae.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `7.8581 deg`
- IFAN without silences mean RMSAE: `7.0755 deg`
- Baseline with silences mean RMSAE: `7.7387 deg`
- Baseline without silences mean RMSAE: `7.1201 deg`
- Delta vs baseline with silences: `+0.1195 deg`
- Delta vs baseline without silences: `-0.0445 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `5.4455 deg`
- IFAN without silences mean RMSAE: `5.4777 deg`
- Baseline with silences mean RMSAE: `5.9252 deg`
- Baseline without silences mean RMSAE: `6.1232 deg`
- Delta vs baseline with silences: `-0.4796 deg`
- Delta vs baseline without silences: `-0.6455 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `10.1375 deg`
- IFAN without silences mean RMSAE: `8.3455 deg`
- Baseline with silences mean RMSAE: `9.0915 deg`
- Baseline without silences mean RMSAE: `7.9956 deg`
- Delta vs baseline with silences: `+1.0459 deg`
- Delta vs baseline without silences: `+0.3499 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `11.8516 deg`
- IFAN without silences mean RMSAE: `9.9599 deg`
- Baseline with silences mean RMSAE: `11.1009 deg`
- Baseline without silences mean RMSAE: `8.8365 deg`
- Delta vs baseline with silences: `+0.7507 deg`
- Delta vs baseline without silences: `+1.1234 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
