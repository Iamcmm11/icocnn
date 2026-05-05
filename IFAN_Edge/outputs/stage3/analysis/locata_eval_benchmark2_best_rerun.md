# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314/checkpoints/best_rmsae.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `7.4228 deg`
- IFAN without silences mean RMSAE: `6.7591 deg`
- Baseline with silences mean RMSAE: `7.7387 deg`
- Baseline without silences mean RMSAE: `7.1201 deg`
- Delta vs baseline with silences: `-0.3158 deg`
- Delta vs baseline without silences: `-0.3610 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `5.0884 deg`
- IFAN without silences mean RMSAE: `5.1980 deg`
- Baseline with silences mean RMSAE: `5.9252 deg`
- Baseline without silences mean RMSAE: `6.1232 deg`
- Delta vs baseline with silences: `-0.8367 deg`
- Delta vs baseline without silences: `-0.9252 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `9.8438 deg`
- IFAN without silences mean RMSAE: `8.7194 deg`
- Baseline with silences mean RMSAE: `9.0915 deg`
- Baseline without silences mean RMSAE: `7.9956 deg`
- Delta vs baseline with silences: `+0.7523 deg`
- Delta vs baseline without silences: `+0.7238 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `11.0712 deg`
- IFAN without silences mean RMSAE: `8.8576 deg`
- Baseline with silences mean RMSAE: `11.1009 deg`
- Baseline without silences mean RMSAE: `8.8365 deg`
- Delta vs baseline with silences: `-0.0297 deg`
- Delta vs baseline without silences: `+0.0212 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
