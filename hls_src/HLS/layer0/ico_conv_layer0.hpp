#ifndef ICO_CONV_LAYER0_HPP
#define ICO_CONV_LAYER0_HPP

#include <cmath>

/*
Layer0 代码结构（简化分层视图）
================================
L0: conv_ico_layer0（顶层）
  L1-1: get_kernel
    L2: 将紧凑的 7 邻域权重展开为 3x3 卷积核
  L1-2: 按时间步处理（t in [0, TIME_STEPS)）
    L2-1: 从 input[t] 提取单帧
    L2-2: pad_ico
      L3: smooth_vertices（对填充前数据做极点平滑）
      L3: 按索引表重排并写入极点值
    L2-3: 将填充后的图表重塑为二维布局
    L2-4: 将 kernel/bias 拉平以便卷积循环使用
    L2-5: 在有效输出区域直接卷积（无 conv_output 缓冲）
    L2-6: 在 output[t] 上原地极点平滑（无 output_frame 缓冲）

辅助函数
--------
- smooth_vertices: 对两个极点顶点做 5 邻域均值平滑
- clean_vertices: 清零极点顶点（作为辅助函数保留）
- pad_ico: 平滑 + 基于索引的填充/重排
- get_kernel: 由紧凑权重构建完整 3x3 卷积核
- conv2d_3x3: 通用二维卷积辅助函数（当前顶层路径未使用）
*/

// 配置参数（与 PyTorch 中第一层 IcoConv 对齐）
#define R_LEVEL     2
#define H           4   // 2^R_LEVEL
#define W           8   // 2^(R_LEVEL+1)
#define CHARTS      5
#define TIME_STEPS  52

#define CIN         1
#define COUT        32
#define RIN         1
#define ROUT        6

#define H_PADDED    (H + 2)
#define W_PADDED    (W + 2)

#define KERNEL_H    3
#define KERNEL_W    3

// HLS pragma 使用的输出通道并行因子
#define OC_PAR_FACTOR 2

typedef float data_t;

// 通过邻域平均对两个极点顶点做平滑。
void smooth_vertices(
    data_t input[CIN][RIN][CHARTS][H][W],
    data_t output[CIN][RIN][CHARTS][H][W]
);

// 清零极点顶点（为完整性与可读性保留的辅助函数）。
void clean_vertices(
    data_t input[CHARTS][H][W],
    data_t output[CHARTS][H][W]
);

// 结合重排索引表的二十面体填充。
// 该函数也会执行极点平滑的预处理。
void pad_ico(
    data_t input[CIN][RIN][CHARTS][H][W],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[RIN][CHARTS][H_PADDED][W_PADDED]
);

// layer0 ConvIco 顶层函数。
void conv_ico_layer0(
    data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W],
    const data_t weight[COUT][CIN][RIN][7],
    const data_t bias[COUT],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W]
);

// 根据紧凑 7 邻域权重和索引映射构建完整 3x3 卷积核。
void get_kernel(
    const data_t weight[COUT][CIN][RIN][7],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    data_t kernel[COUT][ROUT][CIN][RIN][KERNEL_H][KERNEL_W]
);

// 在重塑后张量上的通用标准二维卷积辅助函数。
void conv2d_3x3(
    data_t input[(CIN*RIN)][(CHARTS*H_PADDED)][W_PADDED],
    const data_t kernel[(COUT*ROUT)][(CIN*RIN)][KERNEL_H][KERNEL_W],
    const data_t bias[COUT*ROUT],
    data_t output[(COUT*ROUT)][(CHARTS*H_PADDED)][W_PADDED]
);

#endif // 头文件保护结束
