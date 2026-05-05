# Layer2-5中间层调试指南

本文档对应 `layer2-5` 共享 ConvIco 基础验证框架。

## 新增目录

```text
hls_src/HLS/layer2-5/        # Layer2-5 共享 C/HLS 代码
tools/layer2-5/              # Layer2-5 Python 脚本
hls_testdata/layer2-5/       # Layer2-5 测试数据
```

## 核心思路

`layer2-5` 在当前网络中属于重复空间卷积 block，空间尺寸一致、通道规模一致，因此不再分别写 4 套独立验证代码，而是采用一套共享实现，通过 `--layer 2/3/4/5` 选择具体层的数据。

## 运行顺序（PowerShell，仓库根目录）

1. 生成 `layer2-5` testdata

```powershell
python .\tools\layer2-5\generate_layer2_5_testdata.py `
  --model models\1sourceTracking_icoCNN_robot_K4096_r2_model.bin `
  --layer0-input hls_testdata\layer0\input_rearranged.npy `
  --out-dir hls_testdata\layer2-5 `
  --layers 2,3,4,5 `
  --time-steps 52
```

2. 导出某一层的 Python 中间层（以 layer2 为例）

```powershell
python .\tools\layer2-5\debug_layer2_5_intermediate.py --layer 2
```

3. 编译 C 端

```powershell
cd .\hls_src\HLS\layer2-5
.\build_layer2_5.bat
cd ..\..\..
```

4. 运行某一层的 C 端完整验证

```powershell
cd .\hls_src\HLS\layer2-5
.\test_ico_conv_layer2_5.exe 2
cd ..\..\..
```

5. 导出某一层的 C 中间层

```powershell
cd .\hls_src\HLS\layer2-5
.\test_ico_conv_layer2_5_debug.exe 2
cd ..\..\..
```

6. 对比 Python/C 中间层

```powershell
python .\tools\layer2-5\compare_intermediate_layer2_5.py --layer 2
```

## 说明

1. 详细英文版说明见 `tools/layer2-5/LAYER2_5_VERIFICATION_GUIDE.md`
2. HLS 脚本也已经准备好，后续若需要综合可在 `hls_src/HLS/layer2-5/` 下直接运行
3. 当前版本用于基础验证，后续可在此基础上继续抽象成 `layer2-5` 的统一参数化 block
