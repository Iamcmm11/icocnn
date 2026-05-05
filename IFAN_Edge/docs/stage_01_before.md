# 阶段 01 开始前

## 阶段目标

阶段 1 的目标是先把 IFAN 的前端工程链路搭起来，只围绕 `PHAT + LMS` 主线产出可供阶段 2 使用的双特征二十面体输入。

## 当时已有条件

- 仓库根目录已经具备在线 `SRP-PHAT` 预处理链路。
- 仓库根目录已经具备 `IcoTempCNN` 的训练与评估基础。
- 麦克风阵列与二十面体卷积相关依赖已经可用。

## 当时缺失内容

- 还没有独立的 `IFAN_Edge/` 工作区。
- 还没有 `SRPLMSIcoMap` 分支实现。
- 还没有统一的 `DualFeatureIcoPreprocessor`。
- 还没有针对阶段 1 的形状检查和导出脚本。

## 计划交付

- `IFAN_Edge/` 工程骨架
- 阶段文档模板
- `SRPPHATIcoMapAdapter`
- `SRPLMSIcoMap`
- `DualFeatureIcoPreprocessor`
- `scripts/check_stage1_shapes.py`
- `scripts/visualize_stage1_features.py`
