# IFAN_C8_R2 Stage-1 HLS 第二阶段数据归档报告

## 1. 整体下一步工作是什么

本阶段的目标是为 `ifan_c8_r2_maba_pre_readout_best` 的 Stage-1 HLS 验证准备真实 PyTorch 侧数据闭环，并把后续 MABA / readout / SoftArgMax 需要的数据一并归档。Stage-1 HLS 边界保持不变：

- 输入：`[2, T, 5, 4, 8]`，当前 `T=6`。
- 输出 golden：`[T, 8, 6, 5, 2, 4]`。
- 对齐节点：`debug["final_head_logits"]`，也就是 FeatureMABA/channel readout 之前的 pre-readout feature tensor。
- 后续 MABA 输入：同一个 `final_head_logits`。
- 后续 MABA 输出：`debug["pre_readout_refined_logits"]`，shape 同为 `[6, 8, 6, 5, 2, 4]`。

## 2. 本阶段做了什么

新增导出脚本：

```bash
/home/cmm/miniconda3/envs/icocnn/bin/python IFAN_Edge/scripts/export_stage1_hls_golden.py
```

默认读取：

```text
IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_r2_maba_pre_readout_20260518_215408/checkpoints/best_rmsae.pt
IFAN_Edge/outputs/stage1_features/scene_1/dual_maps.npy
```

默认输出目录：

```text
hls_testdata/stage1_ifan_c8_r2/scene_1_t6/
```

已经归档的核心文件：

- `stage1_input.npy/txt`：HLS Stage-1 输入，shape `[2, 6, 5, 4, 8]`。
- `final_head_logits.npy/txt`：PyTorch Stage-1 golden，shape `[6, 8, 6, 5, 2, 4]`。
- `pre_readout_refined_logits.npy/txt`：MABA refined 后参考输出，不作为当前 Stage-1 HLS 对齐目标。
- `stage1_weights.npz` 和 `weights/*.npy/*.txt`：HLS `IfanStage1Weights` 对应权重。
- `stage1_geometry.npz` 和 `geometry/*.npy/*.txt`：`reorder_*`、`kernel_idx_*`、pool neighbor 索引。
- `stage1_debug_tensors.npz`：PyTorch debug 中间节点，用于后续分模块定位误差。
- `maba_weights.npz` 和 `maba/weights/*.npy/*.txt`：FeatureMABA 的全部线性层、depthwise conv、LayerNorm 权重。
- `maba_debug_tensors.npz` 和 `maba/tensors/*.npy/*.txt`：FeatureMABA 手工拆解出的逐步 golden。
- `readout_weights.npz` 和 `post_maba/weights/*.npy/*.txt`：channel readout 权重。
- `post_maba_tensors.npz` 和 `post_maba/tensors/*.npy/*.txt`：MABA 后的 readout、region max、CleanVertices、SoftArgMax 参考。
- `manifest.json`：checkpoint、输入文件、shape、dtype、min/max、导出环境、兼容加载说明。

脚本为了避开当前 Linux 环境中 `gpuRIR` 初始化 CUDA 的问题，没有导入 `ifan_edge` 顶层包，而是轻量加载 `models/placeholders.py`。checkpoint 中 pre-readout MABA 的历史键名为 `map_refiner.*`，当前代码中为 `feature_refiner.*`；脚本已做兼容映射，并在 manifest 中记录。

## 3. 当前验证结果

导出脚本运行通过：

```text
Input shape: (2, 6, 5, 4, 8)
Golden shape: (6, 8, 6, 5, 2, 4)
Final logits min/max: -2.33815 / 2.57942
MABA output shape: (6, 8, 6, 5, 2, 4)
MABA output min/max: -5.21492 / 5.88773
```

manifest/NPZ 复查通过：

- `stage1_input.npy`：`float32`，finite，min/max `0.3328047 / 1.0`。
- `final_head_logits.npy`：`float32`，finite，min/max `-2.3381534 / 2.5794206`。
- `pre_readout_refined_logits.npy`：`float32`，finite，min/max `-5.2149215 / 5.8877311`。
- `stage1_weights.npz` 关键 shape：
  - `phat_stem_w`: `[8, 1, 1, 7]`
  - `fusion_temporal_w`: `[4, 8, 8, 5]`
  - `final_w`: `[8, 8, 6, 7]`
  - `norm_gamma`: `[16, 8]`
- `stage1_geometry.npz` 关键 shape：
  - `reorder_r2_stem`: `[1, 5, 6, 10]`
  - `reorder_r2_main`: `[6, 5, 6, 10]`
  - `reorder_r1`: `[6, 5, 4, 6]`
  - `kernel_idx_stem`: `[8, 6, 1, 1, 9, 4]`
  - `kernel_idx_main`: `[8, 6, 8, 6, 9, 4]`
- `maba_weights.npz` 关键 shape：
  - `in_proj_weight`: `[16, 8]`
  - `dw_conv_weight`: `[16, 1, 3]`
  - `state_proj_weight`: `[16, 16]`
  - `state_back_weight`: `[16, 8]`
  - `out_proj_weight`: `[8, 16]`
- `maba_debug_tensors.npz` 关键 shape：
  - `input_positions`: `[240, 6, 8]`
  - `in_proj_out`: `[240, 6, 16]`
  - `alpha`: `[240, 6, 8]`
  - `state_sequence`: `[240, 6, 8]`
  - `delta`: `[6, 8, 6, 5, 2, 4]`
  - `output`: `[6, 8, 6, 5, 2, 4]`
- `post_maba_tensors.npz` 关键 shape：
  - `channel_readout_logits`: `[6, 1, 6, 5, 2, 4]`
  - `region_max_logits`: `[6, 5, 2, 4]`
  - `region_argmax_idx`: `[6, 5, 2, 4]`
  - `softargmax_input`: `[6, 5, 2, 4]`
  - `softargmax_prob`: `[6, 5, 2, 4]`
  - `softargmax_indexes`: `[3, 5, 2, 4]`
  - `coords`: `[6, 3]`

MABA 复查结果：

- `maba/tensors/output.npy` 与 `pre_readout_refined_logits.npy` 最大差异：`0.0`。
- `maba/tensors/input.npy + maba/tensors/delta.npy` 与 `maba/tensors/output.npy` 最大差异：`0.0`。
- `softargmax_prob` 每帧概率和范围：`1.0 / 1.000000238418579`。
- 由 `softargmax_prob` 和 `softargmax_indexes` 重建 `coords.npy` 的最大差异：`5.960464477539063e-08`。

当前 native HLS smoke 也通过：

```bash
cd hls_src/HLS/stage1_ifan_c8_r2
make clean && make run
```

输出：

```text
Output shape: [6, 8, 6, 5, 2, 4]
PASS
```

## 4. 仍存在的问题

- `IFAN_Edge/scripts/check_stage1_shapes.py` 在当前 Linux 环境会触发 `gpuRIR` CUDA 初始化失败：
  - 普通运行：`cuRAND: 203 /home/cmm/gpuRIR/src/gpuRIR_cuda.cu 1046`
  - `CUDA_VISIBLE_DEVICES=` 运行：`GPUassert: no CUDA-capable device is detected /home/cmm/gpuRIR/src/gpuRIR_cuda.cu 1037`
- 当前 HLS C++ testbench 仍使用合成权重/合成索引；本阶段只完成真实数据归档，还没有把 `scene_1_t6` 数据读入 testbench 做全链路数值对齐。
- 当前 Stage-1 HLS 对齐目标仍是 MABA 前的 `final_head_logits`；MABA/readout/SoftArgMax 数据已经作为后续阶段 golden 归档。

## 5. 再下一步工作是什么

1. 在 HLS testbench 中读取 `hls_testdata/stage1_ifan_c8_r2/scene_1_t6/stage1_input.txt`、`weights/*.txt`、`geometry/*.txt`。
2. 将读取结果填入 `IfanStage1Weights`、`reorder_r2_stem`、`reorder_r2_main`、`reorder_r1`、`kernel_idx_stem`、`kernel_idx_main`。
3. 先做模块级对齐：stem、residual、shared attention、pool、fusion block、final block。
4. 再做 `ifan_stage1_top` 与 `final_head_logits` 的整体误差统计，记录 max error、RMSE 和 shape。
5. Stage-1 稳定后，按 `maba/tensors` 的顺序实现 FeatureMABA：position flatten -> in_proj -> depthwise causal conv -> LayerNorm -> state_proj/gate -> state scan -> state_back -> out_proj -> residual add。
6. MABA 稳定后，接入 `post_maba/weights` 与 `post_maba/tensors` 验证 channel readout、region max、CleanVertices、SoftArgMax。
7. 数值稳定后再推进定点位宽 sweep 和 HLS `csim/csynth`。
