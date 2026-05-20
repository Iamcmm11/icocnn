# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_r2_map_maba_20260516_154535/checkpoints/best_rmsae.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `7.7562 deg`
- IFAN without silences mean RMSAE: `6.8584 deg`
- Baseline with silences mean RMSAE: `7.7387 deg`
- Baseline without silences mean RMSAE: `7.1201 deg`
- Delta vs baseline with silences: `+0.0175 deg`
- Delta vs baseline without silences: `-0.2616 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `4.8651 deg`
- IFAN without silences mean RMSAE: `4.7522 deg`
- Baseline with silences mean RMSAE: `5.9252 deg`
- Baseline without silences mean RMSAE: `6.1232 deg`
- Delta vs baseline with silences: `-1.0600 deg`
- Delta vs baseline without silences: `-1.3709 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `9.9524 deg`
- IFAN without silences mean RMSAE: `8.0200 deg`
- Baseline with silences mean RMSAE: `9.0915 deg`
- Baseline without silences mean RMSAE: `7.9956 deg`
- Delta vs baseline with silences: `+0.8608 deg`
- Delta vs baseline without silences: `+0.0245 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `13.0768 deg`
- IFAN without silences mean RMSAE: `11.1730 deg`
- Baseline with silences mean RMSAE: `11.1009 deg`
- Baseline without silences mean RMSAE: `8.8365 deg`
- Delta vs baseline with silences: `+1.9759 deg`
- Delta vs baseline without silences: `+2.3365 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
