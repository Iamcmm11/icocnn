# layer0-1_details

本目录用于保存 `layer0` 与 `layer1` 每一轮 HLS 跑完后的“关键证据快照”，风格对齐 `layer2-5_details`。

目的：

1. 固定保留前两层在不同结构版本下的阻碍现象与关键指标。
2. 避免后续继续重构后，旧的 `Final II`、`limited memory ports`、`carried dependence` 现象在新日志中消失而无法回溯。
3. 将“完整跑完的 csynth”与“中途超时但已暴露关键瓶颈的 run”都纳入记录。

建议命名：

- `YYYY-MM-DD_HHMMSS_layer0_csynth_key.md`
- `YYYY-MM-DD_HHMMSS_layer1_csynth_key.md`
- `YYYY-MM-DD_HHMMSS_layer1_partial_key.md`

每次至少记录：

1. 本次来源路径：
   `summary.md`、`csynth.rpt`、`vitis_hls.log`
2. 关键指标：
   `Estimated Clock`、`Total Latency`、`BRAM/DSP/FF/LUT`
3. 关键 pipeline 快照：
   至少 2-4 条 `Pipelining result`
4. 阻碍现象：
   `limited memory ports`
   `carried dependence`
   `Final II > 1`
   `critical path`
5. 对下一步结构优化的直接含义

当前已记录：

1. `2026-04-07_175035_layer0_csynth_key.md`
2. `2026-04-07_175100_layer1_partial_key.md`
3. `2026-04-07_190149_layer0_csynth_key.md`
4. `2026-04-07_192256_layer1_partial_key.md`
