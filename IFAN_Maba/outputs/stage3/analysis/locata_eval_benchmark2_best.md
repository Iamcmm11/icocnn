# LOCATA Validation Summary

- checkpoint: `/home/cmm/icocnn/IFAN_Maba/outputs/stage3/ifan_maba_stage3_full40_freqblock_paper_original_maba_20260420_214446/checkpoints/best_rmsae.pt`
- subset: `eval`
- array: `benchmark2`
- tasks: `task1, task3, task5`

## Overall

- IFAN with silences mean RMSAE: `7.5172 deg`
- IFAN without silences mean RMSAE: `6.7147 deg`
- Baseline with silences mean RMSAE: `7.7387 deg`
- Baseline without silences mean RMSAE: `7.1201 deg`

## Per Task

### task1

- count: `13`
- IFAN with silences mean RMSAE: `4.9583 deg`
- IFAN without silences mean RMSAE: `5.0586 deg`
- Baseline with silences mean RMSAE: `5.9252 deg`
- Baseline without silences mean RMSAE: `6.1232 deg`

### task3

- count: `5`
- IFAN with silences mean RMSAE: `9.9412 deg`
- IFAN without silences mean RMSAE: `8.7502 deg`
- Baseline with silences mean RMSAE: `9.0915 deg`
- Baseline without silences mean RMSAE: `7.9956 deg`

### task5

- count: `5`
- IFAN with silences mean RMSAE: `11.7462 deg`
- IFAN without silences mean RMSAE: `8.9852 deg`
- Baseline with silences mean RMSAE: `11.1009 deg`
- Baseline without silences mean RMSAE: `8.8365 deg`

## Paper Reference

- LOCATA is a paper evaluation dataset, not a training dataset.
- Training dataset in the paper: `LibriSpeech train-clean-100`.
- Simulated test dataset in the paper: `LibriSpeech test-clean`.
- Manual table transcription target:
  - with silences: `Table III`
  - without silences: `Table IV`
