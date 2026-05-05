# 阶段 01 完成后

## 当前结论

阶段 1 当前结论固定为：

“工程链路完成，可视化已导出，论文图一致性未确认。”

## 已完成内容

- 已建立独立的 `IFAN_Edge/` 工作区。
- 已实现阶段 1 `PHAT + LMS` 前端主线：
  - `SRPPHATIcoMapAdapter`
  - `SRPLMSIcoMap`
  - `DualFeatureIcoPreprocessor`
- 已保留阶段 1 主线脚本：
  - `scripts/check_stage1_shapes.py`
  - `scripts/visualize_stage1_features.py`

## 当前张量约定

- `PHAT` 单分支张量：`[B, 1, T, 5, H, W]`
- `LMS` 单分支张量：`[B, 1, T, 5, H, W]`
- 双特征张量：`[B, 2, T, 5, H, W]`
- 当 `r = 2` 时，双特征张量为 `[B, 2, T, 5, 4, 8]`
- `channel 0 = PHAT`
- `channel 1 = LMS`

## 已完成验证

- 阶段 1 形状检查通过：
  - `PHAT shape = (2, 1, 3, 5, 4, 8)`
  - `LMS shape = (2, 1, 3, 5, 4, 8)`
  - `Dual shape = (2, 2, 3, 5, 4, 8)`
- `visualize_stage1_features.py --help` 可正常启动。
- 已保留 `stage1_features` 的 4 个固定场景导出结果：
  - `/home/cmm/icocnn/IFAN_Edge/outputs/stage1_features/scene_1`
  - `/home/cmm/icocnn/IFAN_Edge/outputs/stage1_features/scene_2`
  - `/home/cmm/icocnn/IFAN_Edge/outputs/stage1_features/scene_3`
  - `/home/cmm/icocnn/IFAN_Edge/outputs/stage1_features/scene_4`

## 导出产物

每个场景保留以下主线结果：

- `feature_maps_charts.png`
- `feature_maps_projection.png`
- `feature_maps_projection_contrast.png`
- `phat_maps.npy`
- `lms_maps.npy`
- `dual_maps.npy`
- `phat_projection.png`
- `phat_projection_contrast.png`
- `lms_projection.png`
- `lms_projection_contrast.png`

## 收口说明

- 阶段 1 只保留最初 `PHAT + LMS` 前端和 `stage1_features` 输出。
- 当前能够确认的是工程链路已经打通并导出了可视化结果。
- 当前仍不能确认这些结果与论文图是否严格一致。
