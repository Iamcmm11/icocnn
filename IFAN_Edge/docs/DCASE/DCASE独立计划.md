## DCASE Stereo Folded-Azimuth 独立分支计划

### 摘要

新建一条完全独立的 DCASE 定向训练/评估分支，复用冻结的 `ifan_c8_r2_maba_pre_readout_best` 作为初始化权重，但绝不修改、覆盖或 resume 这条已冻结主线。第一版保持当前 IFAN 主干和 3D 坐标输出形式不变，只把输入改成真正的 2 通道 stereo frontend，把训练目标改成水平面上的 folded-azimuth 单位向量，从而尽量贴近 DCASE 的 `DOA error (deg)` 口径，而不引入检测头和距离头。

第一阶段成功标准：在 `datasets/DCASE2025_Task3/locata_like_devtest_strict/manifest_all.csv` 上，整体 `DOA error (deg)` 要低于当前 zero-shot 基线 `44.9631`。

### 关键改动

- 新建一条 DCASE 专用实验线，与现有 Stage-3 LOCATA 主线完全分离。
  - 新输出根目录：`IFAN_Edge/outputs/dcase_stage3`
  - 新配置文件：`IFAN_Edge/configs/dcase_stereo_folded_azimuth_c8_r2_maba_pre_readout.toml`
  - 新训练入口：`IFAN_Edge/scripts/train_dcase_stereo_ifan.py`
  - 新实验标识和输出后缀：`dcase_stereo_folded_azimuth_c8_r2_maba_pre_readout_init_from_frozen`
  - 不复用冻结 run 的 `resume_*` 路径，不写入原输出目录

- 使用冻结模型做初始化，但只作为只读起点。
  - 初始化权重来源：`IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_r2_maba_pre_readout_20260518_215408/checkpoints/best_rmsae.pt`
  - 结构保持一致：`branch_channels=8`，`map_refiner=pre_readout maba`
  - 继续保留已有的 stale checkpoint key 兼容处理
  - 第一版默认全参数微调，不冻结子模块

- 新增 DCASE stereo manifest 数据集和训练流程，不走当前 LOCATA/LibriSpeech 的随机轨迹生成链路。
  - 训练/验证母集：
    - 使用 `datasets/DCASE2025_Task3/locata_like_strict/manifest_all.csv`
    - 只保留 `split in {dev-train-sony, dev-train-tau}`
  - 最终测试集：
    - 使用 `datasets/DCASE2025_Task3/locata_like_devtest_strict/manifest_all.csv`
    - 训练和模型选择过程中绝不使用该集合
  - 内部验证集切分：
    - 固定随机种子 `42`
    - 按 `(original_split, subset)` 分层，其中 `subset in {static_single_source, moving_single_source}`
    - 每个桶留出 `10%` 做验证，且每个非空桶至少留 `1` 条
    - 剩余样本用于训练
  - 音频预处理：
    - 读取 stereo wav
    - 重采样 `24 kHz -> 16 kHz`
    - 窗长/步长：`K=4096`，`step=3072`
    - 固定 stereo 代理阵列：`[[-0.04, 0, 0], [0.04, 0, 0]]`
  - 标签目标：
    - 直接使用 DCASE metadata 中的 folded azimuth
    - 构造成 `theta = 90 deg`、`phi = dcase_azimuth_to_phi_rad(azimuth)` 的水平面球坐标
    - 再转成水平面 3D 单位向量，继续兼容当前 IFAN 输出头

- 只在这条 DCASE 分支里替换优化目标。
  - 损失仍然使用预测 3D 坐标与目标 3D 坐标之间的 MSE
  - 但目标改为“水平面上的 folded-azimuth 单位向量”
  - 损失掩码：
    - 仅在 metadata 标记为 active 的窗口上计算
    - 额外排除最前面的 `5` 个窗口，与当前因果前端 warm-up 排除策略保持一致
  - 验证主指标：
    - 主指标：`DOA error (deg)`，实现为 active window 上 folded azimuth MAE
    - 辅助诊断：folded azimuth RMSE、当前 horizontal-assumption RMSAE

- 保持 DCASE 评估脚本独立，并作为该分支的标准报告工具。
  - 继续使用 `IFAN_Edge/scripts/evaluate_stage3_dcase2025.py`
  - 面向 DCASE 训练出的 checkpoint 运行
  - 输出 JSON/Markdown 仍放在 `IFAN_Edge/outputs/stage3/analysis`
  - 保留当前文案说明：这是官方 DCASE `DOA error (deg)` 的单源近似，不是完整 leaderboard 提交指标

### 实现说明

- 不要在现有 `IFANTrainingPipeline` 上硬塞 DCASE 训练逻辑，只要那样会把 DCASE 和 LOCATA 的随机轨迹假设绑在一起，就不要走这条路。
  - 第一版推荐做法：新增一套平行的 DCASE 训练模块，尽量复用：
    - `IFANModel` / `IFANModelConfig`
    - `DualFeatureIcoPreprocessor`
    - checkpoint/profile 等可复用辅助逻辑
  - 不通过 `build_random_trajectory_dataset`、`LibriSpeechDataset`、模拟场景缓存来训练 DCASE

- 尽量复用当前 DCASE 评估脚本里已经写好的音频/metadata 解析逻辑。
  - 只有在训练和评估都明显重复时，才提取共享工具模块
  - 不额外发明第二套 DCASE metadata 解析方式

- 在整条新分支里统一使用 `DOA error (deg)` 命名。
  - 训练日志
  - 验证摘要
  - 最终 Markdown 表
  - 比较文档追加表格
  都要统一成同一口径

### 测试计划

- 静态正确性检查
  - 配置文件可以独立加载，并生成 `IFAN_Edge/outputs/dcase_stage3` 下的新输出目录
  - 冻结 checkpoint 可以作为 `ifan_init_checkpoint` 正常加载，不触碰原 run 输出目录
  - train/val/test manifest 切分是确定性的，且 `devtest` 不参与训练和验证
  - 分层切分后，`static_single_source` 和 `moving_single_source` 两类在验证集中都仍然存在，只要对应桶非空

- 冒烟测试
  - CPU 下用 `8` 条样本、`1` 个 epoch 跑通训练闭环
  - CUDA 下用同样小样本跑通训练闭环，并能正常写出 checkpoint/history/summary
  - DCASE 评估脚本能对冒烟 checkpoint 生成 `DOA error (deg)` 为主指标的 JSON/Markdown

- 完整训练验收
  - 第一版完整训练可以正常完成
  - 最优 checkpoint 依据内部验证 `DOA error (deg)` 选择
  - 最终在 `locata_like_devtest_strict` 上评估并生成报告
  - 验收阈值：整体 `DOA error (deg) < 44.9631`

### 假设与默认选择

- 第一版明确不做事件检测、距离预测，也不做完整官方 `F-score (20°/1)`。
- 第一版明确不修改冻结的 `ifan_c8_r2_maba_pre_readout_best` 架构、权重、日志和输出目录。
- 第一版保留现有 IFAN 3D 输出头，不新增 azimuth-only head，而是通过“水平面目标向量”实现 folded-azimuth 训练。
- 第一版训练数据使用 `strict` speech-only 单声源过滤集，因为它最接近当前 LOCATA 迁移目标。
- 由于 `strict` 口径下 moving 训练样本很少，第一版主要目标是先把输入口径和训练目标对齐，而不是把 DCASE stereo 专用模型一次做到最终形态。
