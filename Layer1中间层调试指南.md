# Layer1中间层调试指南

本指南对应 `layer1` 验证链路（Python 导出 + C 验证 + 中间层对比）。

## 新增目录

```text
hls_src/layer1/                    # Layer1 C/HLS 代码与测试
tools/layer1/                      # Layer1 Python 脚本
hls_testdata/layer1/               # Layer1 测试数据
```

## 脚本说明

1. `tools/layer1/generate_layer1_testdata.py`
- 从模型生成 `layer1` 的输入/输出、权重、索引表。

2. `tools/layer1/debug_layer1_intermediate.py`
- 导出 Python 端 frame0 的中间层文件（`py_*.txt`）。

3. `hls_src/layer1/test_ico_conv_layer1_debug.cpp`
- 导出 C 端 frame0 中间层文件（`cpp_*.txt`）。

4. `tools/layer1/compare_intermediate_layer1.py`
- 对比 `py_*.txt` 与 `cpp_*.txt`。

## 运行顺序（PowerShell，仓库根目录）

1. 生成 layer1 testdata

```powershell
python .\tools\layer1\generate_layer1_testdata.py `
  --model models\1sourceTracking_icoCNN_robot_K4096_r2_model.bin `
  --layer0-input hls_testdata\layer0\input_rearranged.npy `
  --out-dir hls_testdata\layer1 `
  --time-steps 52
```

2. 导出 Python 中间层

```powershell
python .\tools\layer1\debug_layer1_intermediate.py
```

3. 编译 C 端

```powershell
cd .\hls_src\layer1
.\build_layer1.bat
cd ..\..
```

4. 运行 C 全量验证

```powershell
cd .\hls_src\layer1
.\test_ico_conv_layer1.exe
cd ..\..
```

5. 运行 C 中间层导出

```powershell
cd .\hls_src\layer1
.\test_ico_conv_layer1_debug.exe
cd ..\..
```

6. 对比 Python/C 中间层

```powershell
python .\tools\layer1\compare_intermediate_layer1.py
```

## 备注

1. 详细英文版说明见：`tools/layer1/LAYER1_VERIFICATION_GUIDE.md`
2. 当前 `layer1` C 实现优先保证验证可读性，后续可再逐步加 HLS pragma 做资源/时序优化。
