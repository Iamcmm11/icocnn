# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_r3_paper_original_20260506_220735/checkpoints/best_rmsae.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `8.5871 deg`
- IFAN without silences mean RMSAE: `7.5825 deg`
- Baseline with silences mean RMSAE: `7.5956 deg`
- Baseline without silences mean RMSAE: `7.0124 deg`
- Delta vs baseline with silences: `+0.9914 deg`
- Delta vs baseline without silences: `+0.5701 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `6.0880 deg`
- IFAN without silences mean RMSAE: `5.8945 deg`
- Baseline with silences mean RMSAE: `5.5723 deg`
- Baseline without silences mean RMSAE: `5.7709 deg`
- Delta vs baseline with silences: `+0.5157 deg`
- Delta vs baseline without silences: `+0.1236 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `11.0955 deg`
- IFAN without silences mean RMSAE: `9.0824 deg`
- Baseline with silences mean RMSAE: `8.9686 deg`
- Baseline without silences mean RMSAE: `7.6868 deg`
- Delta vs baseline with silences: `+2.1269 deg`
- Delta vs baseline without silences: `+1.3957 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `12.5763 deg`
- IFAN without silences mean RMSAE: `10.4713 deg`
- Baseline with silences mean RMSAE: `11.4833 deg`
- Baseline without silences mean RMSAE: `9.5658 deg`
- Delta vs baseline with silences: `+1.0929 deg`
- Delta vs baseline without silences: `+0.9054 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
