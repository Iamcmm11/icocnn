# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/stage3/ifan_stage3_full40_lc_reference_hw2_20260428_102726/checkpoints/best_rmsae.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `7.8793 deg`
- IFAN without silences mean RMSAE: `7.2704 deg`
- Baseline with silences mean RMSAE: `7.7387 deg`
- Baseline without silences mean RMSAE: `7.1201 deg`
- Delta vs baseline with silences: `+0.1406 deg`
- Delta vs baseline without silences: `+0.1504 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `5.2566 deg`
- IFAN without silences mean RMSAE: `5.4251 deg`
- Baseline with silences mean RMSAE: `5.9252 deg`
- Baseline without silences mean RMSAE: `6.1232 deg`
- Delta vs baseline with silences: `-0.6686 deg`
- Delta vs baseline without silences: `-0.6980 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `10.0523 deg`
- IFAN without silences mean RMSAE: `8.5196 deg`
- Baseline with silences mean RMSAE: `9.0915 deg`
- Baseline without silences mean RMSAE: `7.9956 deg`
- Delta vs baseline with silences: `+0.9608 deg`
- Delta vs baseline without silences: `+0.5240 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `12.5251 deg`
- IFAN without silences mean RMSAE: `10.8190 deg`
- Baseline with silences mean RMSAE: `11.1009 deg`
- Baseline without silences mean RMSAE: `8.8365 deg`
- Delta vs baseline with silences: `+1.4242 deg`
- Delta vs baseline without silences: `+1.9825 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
