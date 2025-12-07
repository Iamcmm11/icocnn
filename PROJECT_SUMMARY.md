# IcoConv HLS 实现 - 项目总结

## 已生成文件清单

### 1. Python 数据生成脚本
- `generate_layer0_data.py` - 从 PyTorch 模型导出第一层测试数据

### 2. HLS C++ 源代码（hls_src/ 目录）
- `ico_conv_layer0.hpp` - IcoConv Layer 0 头文件（配置、函数声明）
- `ico_conv_layer0.cpp` - IcoConv Layer 0 实现（核心算法）
- `test_ico_conv.cpp` - 测试主程序（加载数据、运行、对比结果）
- `utils.hpp` - 工具函数（文件读取、误差计算）
- `Makefile` - 编译脚本

### 3. 文档
- `HLS_VERIFICATION_GUIDE.md` - 详细的验证指南

### 4. 快速启动脚本
- `run_verification.bat` - Windows 一键验证脚本

## 核心实现模块

### 模块 1：CleanVertices
**功能：** 将 icosahedral 网格的顶点位置清零  
**输入：** `[CHARTS, H, W]`  
**输出：** `[CHARTS, H, W]`  
**操作：** 设置 `[c, 0, 0]` 和 `[c, 0, H]` 为 0

### 模块 2：PadIco
**功能：** icosahedral 网格的拓扑 padding  
**输入：** `[RIN, CHARTS, H, W]`  
**输出：** `[RIN, CHARTS, H+2, W+2]`  
**操作：** 通过查表 `reorder_idx` 重排元素

### 模块 3：get_kernel
**功能：** 从压缩权重构造完整卷积核  
**输入：** `weight[COUT, CIN, RIN, 7]` + `kernel_expansion_idx`  
**输出：** `kernel[COUT, ROUT, CIN, RIN, 3, 3]`  
**操作：** 索引查表展开，并清零特定位置

### 模块 4：conv2d_3x3
**功能：** 标准 2D 卷积（padding=1）  
**输入：** `[CIN*RIN, CHARTS*H_PADDED, W_PADDED]`  
**输出：** `[COUT*ROUT, CHARTS*H_PADDED, W_PADDED]`  
**操作：** 标准卷积计算，边界检查

### 模块 5：conv_ico_layer0（主函数）
**功能：** 完整的 IcoConv Layer 0 前向传播  
**流程：**
1. 预计算卷积核
2. 逐帧处理（103 帧）
3. 每帧：PadIco → Reshape → Conv2D → Reshape → CleanVertices

## 数据流图

```
输入 [103, 1, 1, 5, 4, 8]
    ↓
提取单帧 [1, 1, 5, 4, 8]
    ↓
PadIco [1, 5, 6, 10]  (通过 reorder_idx 查表)
    ↓
Reshape → [1, 30, 10]  (展平为 2D)
    ↓
Conv2D (kernel: [192, 1, 3, 3])
    ↓
Reshape → [32, 6, 5, 4, 8]  (恢复 icosahedral)
    ↓
去 padding [32, 6, 5, 4, 8]
    ↓
CleanVertices
    ↓
输出单帧 [32, 6, 5, 4, 8]
    ↓
重复 103 次 → 最终输出 [103, 32, 6, 5, 4, 8]
```

## 关键设计决策

### 1. 为什么用查表而不是实时计算？
- icosahedral 网格的拓扑关系固定
- 预计算 `reorder_idx` 和 `kernel_expansion_idx` 可以：
  - 简化 HLS 实现（避免复杂的索引计算）
  - 提高性能（BRAM 查表比实时计算快）
  - 易于验证（与 PyTorch 一致）

### 2. 为什么分帧处理而不是批处理？
- 时间维度（103 帧）太大，无法一次性存储所有中间结果
- 逐帧处理降低内存需求
- 易于流水线化（后续优化）

### 3. 为什么用浮点而不是定点？
- 初期验证阶段，浮点便于调试和对比
- 后续可以无缝切换到 `ap_fixed<16,8>` 等定点类型
- 只需修改 `typedef float data_t;` 这一行

## 验证流程

```
┌─────────────────────────────────────────┐
│ 步骤 1：Python 生成数据                  │
│ - 运行 PyTorch IcoConv Layer 0           │
│ - 保存输入、权重、索引、输出              │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 步骤 2：C++ 编译                         │
│ - g++ 编译所有 .cpp 文件                 │
│ - 生成可执行文件                          │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│ 步骤 3：C++ 验证运行                      │
│ - 读取测试数据                            │
│ - 执行 conv_ico_layer0()                 │
│ - 对比输出与 PyTorch 参考值               │
│ - 计算 Max Error 和 RMSE                 │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────▼──────────┐
        │ Max Error < 1e-3?  │
        └─────────┬──────────┘
         Yes │         │ No
    ┌────────▼──┐  ┌──▼────────┐
    │ PASS ✓    │  │ FAIL ✗    │
    │ 进入 HLS  │  │ 调试 C++  │
    └───────────┘  └───────────┘
```

## 资源占用分析

### 当前配置（r=2, TIME_STEPS=103, COUT=32）

**内存需求：**
- 输入：`103 × 1 × 1 × 5 × 4 × 8 × 4 bytes = 66 KB`
- 输出：`103 × 32 × 6 × 5 × 4 × 8 × 4 bytes = 1.5 MB`
- 权重：`32 × 1 × 1 × 7 × 4 bytes = 0.9 KB`
- 卷积核：`32 × 6 × 1 × 1 × 3 × 3 × 4 bytes = 6.9 KB`
- 索引表：
  - `reorder_idx`: `1 × 5 × 6 × 10 × 4 bytes = 1.2 KB`
  - `kernel_expansion_idx`: `32 × 6 × 1 × 1 × 9 × 4 × 4 bytes = 27 KB`

**总内存：** ~1.6 MB（大部分是输出缓冲）

**计算量（单帧）：**
- Conv2D: `192 × 1 × 30 × 10 × 9 = 518,400 ops`
- 总计（103 帧）：`~53 M ops`

### HLS 优化后预估（Kintex-7 xc7k325t）

| 优化级别 | LUT | BRAM | DSP | 吞吐量 | 延迟 |
|----------|-----|------|-----|--------|------|
| 基础版本 | 15K | 250 | 50 | 1 frame/10ms | 1s |
| Pipeline | 25K | 300 | 80 | 1 frame/1ms | 103ms |
| 完全展开 | 40K | 400 | 150 | 1 frame/100us | 10ms |

**推荐配置：** Pipeline 优化，平衡资源与性能

## 下一步工作

### 短期（1-2 周）
- [ ] 运行验证，确保 C++ 实现正确
- [ ] 小规模测试（TIME_STEPS=1）先通过
- [ ] 添加更多调试输出

### 中期（2-4 周）
- [ ] Vivado HLS C Simulation
- [ ] 添加 HLS pragma 优化
- [ ] C Synthesis 并分析资源占用
- [ ] C/RTL Co-simulation 验证

### 长期（1-2 月）
- [ ] 实现后续层（Temporal Conv、LayerNorm、Pooling）
- [ ] 整合完整网络
- [ ] 板上测试与性能调优
- [ ] 定点化优化

## 常见错误排查

### 编译错误
```
Error: 'cmath' file not found
→ 正常，VSCode linter 问题，实际编译时会找到
```

### 运行时错误
```
Error: Cannot open file input_rearranged.txt
→ 确保在 hls_src/ 目录下运行，且已生成数据
```

### 验证失败
```
Max Error: 10.5
→ 检查数据读取顺序（多维数组索引）
→ 检查索引表是否正确读取
```

## 联系与支持

如果遇到问题：
1. 检查 `HLS_VERIFICATION_GUIDE.md` 中的常见问题
2. 对比 Python 和 C++ 的中间结果
3. 添加调试输出定位问题

---

**版本：** 1.0  
**创建日期：** 2025-12-07  
**作者：** AI Assistant  
**项目：** IcoConv Layer 0 HLS 实现
