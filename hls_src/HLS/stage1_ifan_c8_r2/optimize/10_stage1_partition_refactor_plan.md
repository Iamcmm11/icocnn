# Stage-1 分区重构执行计划 10

日期：2026-05-31

## 1. 判断结论

C8_R2 在双特征前端之后的 fusion-head 主干，与已有 `layer2-5` 硬件线是同型 R1 ConvIco：

```text
ConvIco(Cin=Cout, Rin=Rout=6, H=2, W=4)
```

差异主要是规模参数：

| 路线 | 通道/时间 |
|---|---|
| baseline `layer2-5` | `C=32`, `T=52` |
| IFAN C8_R2 fusion head | `C=8`, `T=6` |

因此，当前不应在 `stage1_ifan_c8_r2` 下继续切一套新的主干架构。后续主干应复用并参数化 `hls_src/HLS/layer2-5`，本目录只保留双特征前端、FeatureMABA、post-MABA 等独立切片。

## 2. 新目录边界

```text
stage1_ifan_c8_r2/
  frontend_dual_feature/   当前活跃双 PHAT/LMS 前端 top
  full_stage1_legacy/      旧完整 ifan_stage1_top，仅回归/追溯
  feature_maba/            pre-readout FeatureMABA 切片
  post_maba/               post-MABA readout/SoftArgMax 切片
  optimize/                阶段记录
```

## 3. 当前默认 HLS 边界

默认 top 切换为：

```text
ifan_dual_frontend_top
```

输入：

```text
[2, T, 5, 4, 8]
```

输出：

```text
[T, 8, 6, 5, 4, 8]
```

该输出是 pre-pooling fused R2 feature。后续 PoolIco 与 R1 fusion/final 主干不再作为本目录默认综合对象。

## 4. 后续执行计划

1. 保留 `frontend_dual_feature` 作为前端双特征切片，用于 PS/PL 分工评估和必要的前端 smoke。
2. 在 `layer2-5` 中增加参数化配置，至少覆盖 `C=8,T=6`，形成 C8_R2 fusion/final ConvIco 复用版本。
3. 对参数化后的 `layer2-5` 跑 native、HLS `csim`、HLS `synth`，与原 `C=32,T=52` 报告形成对照。
4. 将 `feature_maba` 作为下一阶段主要 PL 创新候选，单独跑 native、HLS `csim`、HLS `synth`。
5. `post_maba` 暂时保持独立，不并入默认 HLS top；等 MABA 资源闭合后再决定是否作为 PL head。
6. 旧完整 `ifan_stage1_top` 不再作为默认综合目标，只保留在 `full_stage1_legacy/` 中用于数值回归和历史证据。

## 5. 本轮重构产物

本轮已完成：

- 创建 `frontend_dual_feature/`，新增 `ifan_dual_frontend_top`。
- 移动旧完整 top 到 `full_stage1_legacy/`。
- 移动 FeatureMABA 到 `feature_maba/`。
- 移动 post-MABA 到 `post_maba/`。
- 更新 `build.bat`、`Makefile`、`run_hls.tcl`、`run_hls.bat`、`run_hls_eval.ps1` 默认入口。
- 更新根目录 `README.md`。

本轮验证口径：

- 默认前端 native smoke 通过。
- 旧完整 Stage-1 native 回归通过。
- FeatureMABA native 回归通过。
- post-MABA native 回归通过。
