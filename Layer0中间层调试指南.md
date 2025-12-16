# Layer0 中间层调试指南

## 目标

对齐 Python (PyTorch) 和 C++ (HLS) 的 Layer0 中间层计算，精确定位误差来源。

## 数据流程

```
输入 [1, 1, 5, 4, 8]
    ↓
CleanVertices/SmoothVertices
    ↓
PadIco → [1, 5, 6, 10]
    ↓
Reshape → [1, 30, 10]
    ↓
Conv2D (kernel [192, 1, 3, 3]) → [192, 30, 10]
    ↓
Reshape → [32, 6, 5, 6, 10]
    ↓
去 Padding → [32, 6, 5, 4, 8]
    ↓
CleanVertices/SmoothVertices → 最终输出 [32, 6, 5, 4, 8]
```

## 步骤 1: 运行 Python 端调试

```bash
cd g:\3DSLED\icocnn
python debug_layer0_intermediate.py
```

**输出目录**: `hls_testdata/layer0/debug_intermediate/`

**输出文件**:
- `py_frame0_input.txt` - 输入
- `py_frame0_after_clean.txt` - CleanVertices 后
- `py_frame0_padded.txt` - PadIco 后
- `py_frame0_reshaped_input.txt` - Reshape 后的输入
- `py_frame0_kernel.txt` - 卷积核
- `py_frame0_conv2d_output.txt` - Conv2D 后
- `py_frame0_reshaped_output.txt` - Reshape 后的输出
- `py_frame0_unpadded.txt` - 去 Padding 后
- `py_frame0_final_output.txt` - 最终输出
- `py_frame0_reference.txt` - 参考输出（来自完整推理）

## 步骤 2: 编译并运行 C++ 端调试

```bash
cd g:\3DSLED\icocnn\hls_src

# 编译
g++ -std=c++11 -O2 -I. -Wall -o test_ico_conv_debug.exe ico_conv_layer0.cpp test_ico_conv_debug.cpp

# 或使用批处理文件
build_debug.bat

# 运行
test_ico_conv_debug.exe
```

**输出目录**: `hls_testdata/layer0/debug_intermediate_cpp/`

**输出文件**:
- `cpp_frame0_input.txt` - 输入
- `cpp_frame0_padded.txt` - PadIco 后
- `cpp_frame0_reshaped_input.txt` - Reshape 后的输入
- `cpp_frame0_final_output.txt` - 最终输出

## 步骤 3: 对比结果

```bash
cd g:\3DSLED\icocnn
python compare_intermediate.py
```

**输出示例**:
```
======================================================================
对比: 1. 输入 [1, 1, 5, 4, 8]
======================================================================
  Python 数据: 160 个值
  C++ 数据:    160 个值

  Python:  Min=0.406098, Max=1.000000, Mean=0.721997
  C++:     Min=0.406098, Max=1.000000, Mean=0.721997

  差异统计:
    Max Error: 0.00000012
    RMSE:      0.00000003
    Mean Abs Error: 0.00000002
  ✓ PASS: 数据完全一致（误差 < 1e-5）
```

## 调试思路

### 如果 Padding 后就有差异

**可能原因**:
1. `reorder_idx` 的索引计算有误
2. C++ 中的数组索引映射逻辑与 Python 不一致

**检查方法**:
```python
# Python 端
reorder_idx = np.load('hls_testdata/layer0/reorder_idx.npy')
print(reorder_idx[0, 0, :3, :3])  # 查看左上角

# C++ 端：在 test_ico_conv_debug.cpp 中添加打印
for (int h = 0; h < 3; h++) {
    for (int w = 0; w < 3; w++) {
        std::cout << reorder_idx[0][0][h][w] << " ";
    }
    std::cout << std::endl;
}
```

### 如果卷积后有差异

**可能原因**:
1. `kernel_expansion_idx` 的使用有误
2. Conv2D 的实现有问题
3. bias 的添加方式不一致

**检查方法**:
- 对比 `py_frame0_kernel.txt` 和实际 C++ 构造的 kernel
- 检查 Conv2D padding 参数（应该是 `padding=(1,1)`）
- 确认 bias 是否正确重复到 192 个通道

### 如果最终输出有差异

**可能原因**:
1. `CleanVertices` / `SmoothVertices` 的实现不一致
2. 顶点位置的 mask 不正确

**检查方法**:
```python
# 检查 mask
layer0 = ConvIco(r=2, Cin=1, Cout=32, Rin=1, smooth_vertices=True)
print(layer0.process_vertices.mask)  # 如果是 CleanVertices
```

## 数据量说明

**单帧数据量**:
- 输入: 1×1×5×4×8 = 160 个值
- Padding 后: 1×5×6×10 = 300 个值
- Reshaped 输入: 1×30×10 = 300 个值
- Conv2D 输出: 192×30×10 = 57,600 个值（较大，只输出前 5 通道）
- 最终输出: 32×6×5×4×8 = 30,720 个值（只输出前 3 通道）

**完整 52 帧数据量**:
- 总输入: 52×160 = 8,320 个值
- 总输出: 52×30,720 = 1,597,440 个值

因此我们只针对**第 0 帧**进行中间层调试，数据量可控且易于查看。

## 文件结构

```
icocnn/
├── debug_layer0_intermediate.py       # Python 端中间层调试
├── compare_intermediate.py            # 对比脚本
├── hls_src/
│   ├── test_ico_conv_debug.cpp        # C++ 端中间层调试
│   ├── build_debug.bat                # 编译脚本
│   └── ...
└── hls_testdata/layer0/
    ├── debug_intermediate/            # Python 输出（py_*.txt）
    └── debug_intermediate_cpp/        # C++ 输出（cpp_*.txt）
```

## 预期结果

理想情况下，Python 和 C++ 的所有中间层输出应该满足：
- **Max Error < 1e-5**: 完全一致
- **Max Error < 1e-3**: 可接受的浮点误差
- **Max Error > 1e-3**: 需要排查实现差异

当前状态：Max Error ≈ 3.7，需要通过中间层调试定位问题！
