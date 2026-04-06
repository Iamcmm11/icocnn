# layer2-5 csynth 关键快照（YYYY-MM-DD HH:MM:SS）

## 1) 来源

- `summary`: `hls_src/hls_reports/<run_dir>/summary.md`
- `csynth`: `hls_src/hls_reports/<run_dir>/conv_ico_layer2_5_csynth.rpt`
- `log`: `hls_src/HLS/layer2-5/layer2_5_hls_prj/sol1/sol1.log`

## 2) 关键指标（极简）

- Target Clock: `<...> ns`
- Estimated Clock: `<...> ns`
- Total Latency: `<...> cycles`
- BRAM_18K: `<...>`
- DSP: `<...>`
- FF: `<...>`
- LUT: `<...>`

## 3) 关键 pipeline 快照（本次）

1. `<module_or_loop_name>`: `Final II = <...>`
2. `<module_or_loop_name>`: `Final II = <...>`
3. `<module_or_loop_name>`: `Final II = <...>`
4. `Loop Constraint Status`: `<...>`

## 4) 论文关键阻碍现象追踪（固定三项）

| 追踪项 | 本次是否直接出现 | 本次对应现象 | 证据来源 |
|---|---|---|---|
| `input_r_load_* due to limited memory ports` | `<是/否>` | `<...>` | `<log/rpt/doc>` |
| `pad_ico_Pipeline_VITIS_LOOP_76_2` | `<是/否>` | `<...>` | `<log/rpt/doc>` |
| `Final II = 27` | `<是/否>` | `<...>` | `<log/rpt/doc>` |

## 5) 历史阻碍锚点（防丢）

1. `input_r_load_* due to limited memory ports`
2. `pad_ico_Pipeline_VITIS_LOOP_76_2`
3. `Final II = 27`

历史文本锚点：`hls_src/layer2-5硬件优化与策略跟踪.md`（第 4.2 节问题 B 与相关阶段记录）。

## 6) 一句话结论

`<本次 1 句总结：最好同时包含 latency、资源、阻碍现象状态>`

