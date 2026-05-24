# IFAN_C8_R2 Stage-1 下一阶段计划

## Summary
下一阶段目标是把当前 README 中“第一阶段结构框架”推进到“真实数据入口闭环”：基于 `ifan_c8_r2_maba_pre_readout_best` checkpoint，导出 Stage-1 HLS 所需的真实输入、真实权重/几何索引、PyTorch golden 输出，并完成 shape/smoke 验证。当前 README 保留为第一阶段报告；本阶段完成后在 `hls_src/HLS/stage1_ifan_c8_r2/NEXT_STAGE_REPORT.md` 写第二阶段报告。

## Tools And Defaults
- 可直接使用：`g++`、`build.bat`、`run_hls.bat`、`vitis_hls 2024.2`、IFAN_Edge 的 `DualFeatureIcoPreprocessor`、`IFANModel(return_debug=True)`、现有 `check_stage1_shapes.py`。
- Python 默认使用：`G:\PostGraduateFile\anaconda\envs\ocr_pdf\python.exe`，因为当前 base Python 缺 `torch`，`Python_anaconda` 环境 numpy DLL 异常；`yolov5` 环境可作为备用。
- 本阶段不以 `csynth` 为验收目标；只做导出、shape 检查、native smoke/build、PyTorch golden 文件生成。
- `make` 缺失，因此 Windows 路径统一使用 `build.bat` 和 `run_hls.bat`。

## Key Changes
- 新增一个导出脚本，建议路径：`IFAN_Edge/scripts/export_stage1_hls_golden.py`。
- 脚本固定读取 checkpoint：
  `IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_r2_maba_pre_readout_20260518_215408/checkpoints/best_rmsae.pt`
- 导出目录固定为：
  `hls_testdata/stage1_ifan_c8_r2/scene_1_t6/`
- 导出内容：
  - `stage1_input.npy/txt`：HLS 输入 `[2, T, 5, 4, 8]`，默认 `T=6`。
  - `final_head_logits.npy/txt`：PyTorch MABA 前输出 `[T, 8, 6, 5, 2, 4]`。
  - `manifest.json`：checkpoint 路径、config、tensor shape、dtype、min/max、导出时间、所用 Python 环境。
  - 权重和索引先导出为 Python-friendly `.npz`，同时预留 HLS text/header 输出格式，避免第一版被 C++ 初始化细节拖慢。
- 不修改 `ifan_stage1_top` 接口：仍保持输入 `[2,T,5,4,8]`、输出 `[T,8,6,5,2,4]`。

## Implementation Details
- 脚本从 checkpoint 的 `training_config` 还原 `IFANModelConfig`，加载 `model_state_dict`，调用 `model.eval()`。
- 输入数据优先用 IFAN_Edge 现有 `DualFeatureIcoPreprocessor` 生成 PHAT/LMS 双特征序列；若本地 LibriSpeech 不可用，则先提供 deterministic synthetic mic batch 作为可复现入口样本，并在 `manifest.json` 明确标注。
- 通过 `model(x, return_debug=True)` 导出：
  - `debug["final_head_logits"]` 作为 Stage-1 golden 输出。
  - 如存在 `debug["pre_readout_refined_logits"]`，只记录到 manifest，不作为本阶段 HLS 对齐目标。
- 几何索引导出从实际 `ConvIco` 模块读取 `kernel_expansion_idx` 和 padding `reorder_idx`；如果某些对象名不一致，脚本应打印可定位的模块清单并失败，不静默生成伪索引。
- 阶段报告 `NEXT_STAGE_REPORT.md` 结构固定为：
  1. 整体下一步工作是什么
  2. 本阶段做了什么
  3. 当前验证结果
  4. 仍存在的问题
  5. 再下一步工作是什么

## Test Plan
- 运行：
  `G:\PostGraduateFile\anaconda\envs\ocr_pdf\python.exe IFAN_Edge/scripts/check_stage1_shapes.py`
- 运行新导出脚本，检查导出文件存在且 manifest 中 shape 为：
  - input: `[2, 6, 5, 4, 8]`
  - golden: `[6, 8, 6, 5, 2, 4]`
- 运行：
  `cd hls_src\HLS\stage1_ifan_c8_r2 && build.bat`
- 运行：
  `test_ifan_stage1.exe`
- 可选运行：
  `run_hls.bat csim`
  仅记录是否能编译/启动，不要求完整数值对齐通过。

## Assumptions
- 本阶段验收重点是“真实 PyTorch 侧数据与 HLS 入口文件准备好”，不是完成 C++ testbench 的全量数值对齐。
- `hls_testdata/stage1_ifan_c8_r2/scene_1_t6/` 作为本阶段默认数据目录。
- 当前 README 视为第一阶段报告，不回写大段内容；第二阶段报告单独新增，保持阶段边界清楚。
