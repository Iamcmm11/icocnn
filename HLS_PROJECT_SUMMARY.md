# Layer0 IcoConv C++ 验证项目总结

## 🎯 项目目标

为 FPGA (Kintex-7 xc7k325t) 部署 IcoTempCNN 的第一层 IcoConv，分三阶段实现:

```
阶段 1: 纯 C++ 顺序执行验证 (当前)
  ├─ 目标: 验证算法逻辑与 PyTorch 完全一致
  ├─ 工具: g++ / MSVC (标准 C++ 编译器)
  ├─ 特点: 不添加 HLS pragma，纯粹的顺序执行
  └─ 验证: 最大误差 < 1e-4

阶段 2: Vivado HLS 综合 (下一步)
  ├─ 导入验证通过的 C++ 代码
  ├─ 添加 #pragma HLS 优化指令
  └─ 生成 IP 核

阶段 3: FPGA 板级部署 (最终目标)
  └─ 集成到完整系统
```

## 📁 文件结构

```
icocnn/
├─ hls_testdata/                    # 测试数据目录 (需从服务器上传)
│   └─ layer0/
│       ├─ input.txt               # 输入 [1,4,10,42]=1680值
│       ├─ weight.txt              # 权重 [32,4,7]=896值
│       ├─ bias.txt                # 偏置 [32]=32值
│       ├─ neighbors.txt           # 邻居索引 [42,7]=294值
│       └─ output.txt              # PyTorch参考输出 [1,32,10,42]=13440值
└─ hls_implementation/              # C++ 验证框架
    ├─ include/                     # 头文件
    │   ├─ ico_types.h              # 类型定义
    │   ├─ ico_conv_layer0.h        # Layer0接口
    │   └─ utils.h                  # 工具函数
    ├─ src/                         # 源代码
    │   ├─ ico_conv_layer0.cpp      # Layer0实现
    │   ├─ utils.cpp                # 工具实现
    │   ├─ test_ico_conv.cpp        # 测试主程序
    │   └─ Makefile                 # 编译脚本
    ├─ 编译并测试.bat               # Windows快捷脚本
    ├─ 数据文件说明.md             # 数据格式说明
    └─ README.md                    # 详细文档
```

## 🎯 项目目标

为 FPGA (Kintex-7 xc7k325t) 部署 IcoTempCNN 的第一层 IcoConv,实现:
1. **功能验证**: HLS C++ 实现与 PyTorch 输出完全一致
2. **资源评估**: 确认 FPGA 资源充足
3. **性能基准**: 为后续优化建立基线

## 🔧 快速开始

### 第一步: 准备数据

从服务器上传 5 个 txt 文件到 `hls_testdata/layer0/`:
- `input.txt` (1680个值)
- `weight.txt` (896个值) 
- `bias.txt` (32个值)
- `neighbors.txt` (294个整数索引)
- `output.txt` (13440个值, PyTorch参考输出)

详细格式见: `hls_implementation/数据文件说明.md`

### 第二步: 运行验证

```bash
# 方式 1: 一键测试 (推荐)
cd hls_implementation
编译并测试.bat

# 方式 2: 手动编译
cd hls_implementation/src
make clean
make
make run
```

### 第三步: 查看结果

看到这个就成功了:
```
✓✓✓ 测试通过! C++ 实现与 PyTorch 一致 ✓✓✓
```

然后就可以开始 Vivado HLS 综合了!

## 📊 Layer0 技术细节

### 网络结构
```
IcoConv Layer0:
  输入:  [B=1, C_in=4,  T=10, V=42]
  权重:  [C_out=32, C_in=4, N=7]
  输出:  [B=1, C_out=32, T=10, V=42]
```

### 计算逻辑
```cpp
for each (batch b, time t, output_channel out_c, vertex v):
    sum = 0
    for each (input_channel in_c, neighbor n):
        neighbor_idx = neighbors[v][n]
        sum += input[b][in_c][t][neighbor_idx] * weight[out_c][in_c][n]
    output[b][out_c][t][v] = ReLU(sum + bias[out_c])
```

### 关键特性
- **邻居卷积**: 每个顶点有 7 个邻居 (包括自己)
- **激活函数**: ReLU
- **数据格式**: NCHW (batch, channel, time, vertex)
- **浮点精度**: float32

## 🎓 代码说明

### 核心文件

#### `ico_types.h`
定义了:
- 基本类型 (`data_t=float`, `index_t=int`)
- 配置宏 (`IN_CHANNELS=4`, `OUT_CHANNELS=32`, 等)
- 索引计算宏 (`INPUT_IDX`, `OUTPUT_IDX`, 等)
- ReLU 激活函数

#### `ico_conv_layer0.cpp`
实现了:
1. `compute_conv_pixel()` - 单像素卷积计算
2. `load_input_tile()` - 输入数据缓存
3. `load_weight_tile()` - 权重数据缓存
4. `ico_conv_layer0()` - 主卷积函数

#### `utils.cpp`
提供了:
- 文件读写 (txt 格式)
- 误差计算 (最大误差、RMSE)
- 统计信息打印

#### `test_ico_conv.cpp`
测试流程:
1. 加载测试数据
2. 运行 HLS 卷积
3. 对比 PyTorch 参考输出
4. 报告误差和通过状态

## ✅ 验证标准

- **误差阈值**: `1e-4`
- **判定**: `max_abs_error < 1e-4` → 通过 ✓
- **输出**: 同时显示最大绝对误差和相对误差 (RMSE)

## 🚀 后续优化方向

### 1. HLS 综合优化

添加 pragma 指令:
```cpp
#pragma HLS PIPELINE II=1           // 循环流水线
#pragma HLS ARRAY_PARTITION ...     // 数组分割
#pragma HLS UNROLL factor=4         // 循环展开
```

### 2. 接口设计

添加 AXI 接口:
```cpp
#pragma HLS INTERFACE m_axi port=input  bundle=gmem0
#pragma HLS INTERFACE m_axi port=output bundle=gmem1
#pragma HLS INTERFACE s_axilite port=return
```

### 3. 量化优化

考虑定点数:
- `ap_fixed<16,8>` 替代 float
- 减少 DSP 和功耗
- 评估精度损失

### 4. 多层集成

后续层级:
- Layer1: 第二个 IcoConv
- Layer2-3: Temporal 卷积
- Layer4: SoftArgMax
- 完整端到端流水线

## 📈 资源估算

### 你的板子 (Kintex-7 xc7k325t)
- **LUT**: 203,800
- **BRAM**: 445
- **DSP**: 840

### Layer0 预估 (未优化)
- **LUT**: ~5,000 (2.5%)
- **BRAM**: ~10 (2.2%)
- **DSP**: ~64 (7.6%)
- **延迟**: ~100K cycles @ 100MHz = 1ms

### 完整网络预估
- **总层数**: ~6-8 层
- **总资源**: 30-50% LUT, 20-30% BRAM, 40-60% DSP
- **吞吐量**: ~10ms/frame (目标)

✓ **结论**: 资源充足,可以部署完整网络!

## 🔍 调试技巧

### 如果测试失败:

1. **检查数据路径**
   ```cpp
   #define DATA_DIR "../hls_testdata/layer0/"
   ```

2. **查看中间输出**
   在 `ico_conv_layer0.cpp` 中添加:
   ```cpp
   printf("Debug: out_c=%d, v=%d, val=%.6f\n", out_c, v, out_val);
   ```

3. **对比 Python 输出**
   运行 `inference_debug.py` 查看 PyTorch 的中间结果

4. **检查邻居索引**
   确保 `neighbors.txt` 加载正确

## 📝 文件清单

| 文件 | 行数 | 功能 |
|------|------|------|
| `generate_layer0_data.py` | ~135 | 生成测试数据 |
| `ico_types.h` | ~56 | 类型和配置 |
| `ico_conv_layer0.h` | ~52 | Layer0 接口 |
| `ico_conv_layer0.cpp` | ~123 | Layer0 实现 |
| `utils.h` | ~46 | 工具接口 |
| `utils.cpp` | ~141 | 工具实现 |
| `test_ico_conv.cpp` | ~137 | 测试主程序 |
| `Makefile` | ~61 | 编译脚本 |
| `README.md` | ~198 | 详细文档 |
| **总计** | **~949** | 约 1000 行代码 |

## 💡 关键设计决策

### 为什么先做 Layer0?
1. **最简单**: 只有卷积 + ReLU,没有池化、归一化
2. **代表性强**: 后续 IcoConv 层结构类似
3. **验证核心**: 验证邻居索引和卷积逻辑

### 为什么用 txt 格式?
1. **可读性**: 方便人工检查
2. **跨平台**: 避免字节序问题
3. **调试友好**: 直接用文本编辑器查看

### 为什么分离头文件?
1. **HLS 友好**: Vivado HLS 推荐的结构
2. **可复用**: 后续层可复用类型定义
3. **清晰**: 接口和实现分离

## 🎉 快速开始

最简单的方式:
```bash
# 在项目根目录运行
run_hls_test.bat
```

看到这个就成功了:
```
✓✓✓ 测试通过! ✓✓✓
```

## 📞 问题排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `Cannot open file` | 路径错误 | 检查 `DATA_DIR` 定义 |
| `编译失败` | 编译器未安装 | 安装 MinGW/MSVC |
| `误差很大` | 索引/计算错误 | 对比 Python 代码 |
| `数据生成失败` | PyTorch 环境 | 检查依赖安装 |

## 📚 参考资料

- **Python 实现**: `acousticTrackingModels.py` - IcoTempCNN 类
- **icoCNN 源码**: `icoCNN-master/icoCNN/icoCNN.py` - ConvIco 实现
- **调试脚本**: `inference_debug.py` - 各层输出对比
- **C 前端**: `c_implementation/` - FFT/GCC-PHAT/SRP 参考

---

**祝你成功部署到 FPGA! 🚀**
