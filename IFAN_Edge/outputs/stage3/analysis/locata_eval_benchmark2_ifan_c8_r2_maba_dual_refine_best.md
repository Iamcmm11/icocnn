# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_r2_maba_dual_refine_20260519_191649/checkpoints/best_rmsae.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `7.7096 deg`
- IFAN without silences mean RMSAE: `6.7878 deg`
- Baseline with silences mean RMSAE: `7.7387 deg`
- Baseline without silences mean RMSAE: `7.1201 deg`
- Delta vs baseline with silences: `-0.0291 deg`
- Delta vs baseline without silences: `-0.3323 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `5.2623 deg`
- IFAN without silences mean RMSAE: `5.3078 deg`
- Baseline with silences mean RMSAE: `5.9252 deg`
- Baseline without silences mean RMSAE: `6.1232 deg`
- Delta vs baseline with silences: `-0.6628 deg`
- Delta vs baseline without silences: `-0.8153 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `10.2160 deg`
- IFAN without silences mean RMSAE: `8.3030 deg`
- Baseline with silences mean RMSAE: `9.0915 deg`
- Baseline without silences mean RMSAE: `7.9956 deg`
- Delta vs baseline with silences: `+1.1245 deg`
- Delta vs baseline without silences: `+0.3075 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `11.5660 deg`
- IFAN without silences mean RMSAE: `9.1205 deg`
- Baseline with silences mean RMSAE: `11.1009 deg`
- Baseline without silences mean RMSAE: `8.8365 deg`
- Delta vs baseline with silences: `+0.4651 deg`
- Delta vs baseline without silences: `+0.2840 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
