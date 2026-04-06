# layer2-5_details

本目录用于保存 `layer2-5` 每一次 `csynth` 后的“关键信息快照”。

目的：

1. 固定保留论文关键证据点（尤其是 pipeline 阻碍现象）。
2. 避免后续重构后，旧 loop 名称或 `Final II` 现象在新日志中消失而无法追溯。
3. 保持每次记录极简、可横向对比。

文件命名建议：

- `YYYY-MM-DD_HHMMSS_csynth_key.md`

每次至少记录以下内容：

1. 本次报告来源路径（`summary.md`、`csynth.rpt`、`sol1.log`）。
2. 关键资源与时延（`Estimated Clock`、`Total Latency`、`BRAM/DSP/FF/LUT`）。
3. 关键 pipeline 快照（至少 2-4 条 `Pipelining result`）。
4. 关键阻碍现象追踪（是否出现、出现在哪个文件、对应原始关键词）。

阻碍现象关键词（固定追踪）：

- `input_r_load_* due to limited memory ports`
- `pad_ico_Pipeline_VITIS_LOOP_76_2`
- `Final II = 27`

备注：

- 某些关键词可能只在历史版本出现，不会在新版本中再次出现。该情况必须记录为“本次未出现 + 历史来源”。

