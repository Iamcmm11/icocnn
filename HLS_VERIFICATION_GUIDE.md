# IcoConv Layer 0 HLS 验证指南

## 项目结构

```
icocnn/
├── hls_src/                        # HLS C++ 源代码
│   ├── ico_conv_layer0.hpp         # IcoConv 头文件
│   ├── ico_conv_layer0.cpp         # IcoConv 实现
│   ├── test_ico_conv.cpp           # 测试主程序
│   ├── utils.hpp                   # 工具函数（文件读取/误差计算）
│   └── Makefile                    # 编译脚本
├── hls_testdata/layer0/            # 测试数据目录（运行 Python 脚本后生成）
│   ├── input_rearranged.txt        # 输入数据
│   ├── weight.txt                  # 权重
│   ├── bias.txt                    # 偏置
│   ├── kernel_expansion_idx.txt    # 卷积核展开索引
│   ├── reorder_idx.txt             # Padding 重排索引
│   ├── output_layer0.txt           # PyTorch 参考输出
│   └── config.txt                  # 配置参数
└── generate_layer0_data.py         # 数据生成脚本
```

## 快速开始

### 步骤 1：生成测试数据

首先运行 Python 脚本生成第一层的测试数据：

```bash
cd g:\3DSLED\icocnn
python generate_layer0_data.py
```

**输出说明：**
- 脚本会创建 `hls_testdata/layer0/` 目录
- 生成所有必要的输入、权重、索引表和参考输出
- 所有数据都会保存为 `.txt` 格式（每行一个值）和 `.npy` 格式

**预期输出：**
```
=== Generating Layer 0 Test Data ===

Input shape: torch.Size([1, 1, 103, 5, 4, 8])
Saved input: shape=(1, 1, 103, 5, 4, 8), size=16480

Layer 0 Info:
  Type: <class 'icoCNN.icoCNN.ConvIco'>
  r=2, Cin=1, Cout=32
  Rin=1, Rout=6

Weight shape: (32, 1, 1, 7)
Saved weight: shape=(32, 1, 1, 7), size=224
...
=== All data saved to hls_testdata/layer0/ ===
```

### 步骤 2：编译 C++ 测试程序

进入 `hls_src` 目录并编译：

```bash
cd hls_src

# 使用 g++ 编译
g++ -std=c++11 -O2 -I. -Wall -o test_ico_conv ico_conv_layer0.cpp test_ico_conv.cpp
```

或者使用 Makefile：

```bash
make
```

### 步骤 3：运行验证

```bash
# Windows
test_ico_conv.exe

# Linux/Mac
./test_ico_conv
```

**预期输出：**
```
=== IcoConv Layer 0 HLS Testbench ===

[1] Loading input data...
Input: size=16480, min=-2.xxx, max=2.xxx, mean=0.xxx

[2] Loading weights and bias...
Weight: size=224, min=-0.xxx, max=0.xxx, mean=0.xxx
Bias: size=32, min=-0.xxx, max=0.xxx, mean=0.xxx

[3] Loading index tables...
Kernel expansion idx size: 6912
Reorder idx size: 180

[4] Preparing arrays...
Arrays prepared successfully.

[5] Running IcoConv Layer 0...
IcoConv Layer 0 finished.

[6] Comparing with reference output...
HLS Output: size=395520, min=-xx.xx, max=xx.xx, mean=x.xx
Reference Output: size=395520, min=-xx.xx, max=xx.xx, mean=x.xx

=== Verification Results ===
Max Error: 0.000xxx
RMSE: 0.000xxx

✓ PASS: HLS output matches PyTorch reference!
```

## 验证标准

- **Max Error < 1e-3**：认为 HLS 实现正确 ✓
- **Max Error < 1e-2**：可接受的浮点误差 ⚠️
- **Max Error > 1e-2**：存在实现问题，需要调试 ✗

## 常见问题

### Q1: 编译错误 "找不到头文件"

**解决方法：**
确保在 `hls_src` 目录下编译，且所有 `.hpp` 文件都在同一目录。

### Q2: 运行时找不到数据文件

**解决方法：**
确保已经运行了 `generate_layer0_data.py`，并且在 `hls_src` 目录下运行测试程序（因为代码中使用相对路径 `../hls_testdata/layer0/`）。

### Q3: 数据大小不匹配

**解决方法：**
检查 `ico_conv_layer0.hpp` 中的配置参数是否与 Python 脚本中的一致：
- `R_LEVEL = 2`
- `CHANNELS = 32`
- `TIME_STEPS = 103`

### Q4: 误差很大（> 1e-2）

**可能原因：**
1. 数据读取顺序错误（检查多维数组的索引顺序）
2. 索引表读取错误（kernel_expansion_idx 或 reorder_idx）
3. 卷积计算逻辑有误

**调试方法：**
1. 先测试小规模数据（修改 TIME_STEPS = 1）
2. 在关键位置添加 `printf` 输出中间结果
3. 对比 Python 和 C++ 的中间层输出

## 下一步：Vivado HLS 综合

成功验证 C++ 实现后，可以进行 HLS 综合：

### 1. 创建 HLS 工程

在 Vivado HLS 中创建新工程，添加：
- 源文件：`ico_conv_layer0.cpp`、`ico_conv_layer0.hpp`
- 测试文件：`test_ico_conv.cpp`、`utils.hpp`
- 顶层函数：`conv_ico_layer0`

### 2. 添加优化指令

在 `ico_conv_layer0.cpp` 中添加 HLS pragma：

```cpp
void conv_ico_layer0(...) {
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS INTERFACE m_axi depth=16480 port=input
    #pragma HLS INTERFACE m_axi depth=395520 port=output
    
    // 优化：展开循环、流水线化
    #pragma HLS PIPELINE II=1
    #pragma HLS DATAFLOW
    ...
}
```

### 3. 运行综合

- C Simulation：验证功能正确性
- C Synthesis：生成 RTL 并查看资源占用
- C/RTL Co-simulation：验证 RTL 正确性
- Export RTL：导出 IP 核

## 性能优化建议

1. **减少时间维度**：如果资源紧张，可以先测试 `TIME_STEPS = 1`
2. **定点化**：将 `data_t` 从 `float` 改为 `ap_fixed<16,8>`
3. **并行化**：使用 `#pragma HLS UNROLL` 展开内层循环
4. **流水线**：使用 `#pragma HLS PIPELINE` 提高吞吐量
5. **数组分割**：使用 `#pragma HLS ARRAY_PARTITION` 减少内存瓶颈

## 资源估算（Kintex-7 xc7k325t）

基于当前实现的初步估算：

| 资源 | 预估占用 | 总量 | 利用率 |
|------|----------|------|--------|
| LUT | 15K-25K | 203K | 8-12% |
| FF | 10K-20K | 407K | 2-5% |
| BRAM | 150-250 | 445 | 34-56% |
| DSP | 50-100 | 840 | 6-12% |

**注意：** 实际资源占用取决于优化程度和综合策略。

## 参考文献

- icoCNN 论文：Cohen et al., "Gauge Equivariant Convolutional Networks and the Icosahedral CNN"
- Vivado HLS 用户手册：UG902
- Xilinx HLS 优化指南：UG1270
