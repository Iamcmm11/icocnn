#ifndef ICO_CONV_LAYER0_HPP
#define ICO_CONV_LAYER0_HPP

#include <cmath>

// ==================== 配置参数 ====================
// 这些参数对应 PyTorch 中的第一层 IcoConv
#define R_LEVEL     2           // icosahedral 分辨率
#define H           4           // 2^R_LEVEL
#define W           8           // 2^(R_LEVEL+1)
#define CHARTS      5           // icosahedral 网格的 charts 数量
#define TIME_STEPS  103         // 时间维度

#define CIN         1           // 输入通道
#define COUT        32          // 输出通道
#define RIN         1           // 输入方向通道（1 或 6）
#define ROUT        6           // 输出方向通道

#define H_PADDED    (H + 2)     // padding 后的高度
#define W_PADDED    (W + 2)     // padding 后的宽度

#define KERNEL_H    3           // 卷积核高度
#define KERNEL_W    3           // 卷积核宽度

// ==================== 数据类型定义 ====================
typedef float data_t;           // 可以改为 ap_fixed<16,8> 做定点化

// ==================== 核心函数声明 ====================

// 1. CleanVertices: 将顶点位置清零
void clean_vertices(
    data_t input[CHARTS][H][W],
    data_t output[CHARTS][H][W]
);

// 2. PadIco: icosahedral padding（通过查表重排）
void pad_ico(
    data_t input[RIN][CHARTS][H][W],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[RIN][CHARTS][H_PADDED][W_PADDED]
);

// 3. ConvIco: icosahedral 卷积（核心）
void conv_ico_layer0(
    // 输入: [1, 103, Cin=1, Rin=1, 5, H, W]
    data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W],
    
    // 权重: [Cout, Cin, Rin, 7] -> 通过 kernel_expansion_idx 展开为 [Cout, Rout, Cin, Rin, 3, 3]
    const data_t weight[COUT][CIN][RIN][7],
    const data_t bias[COUT],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    
    // 输出: [1, 103, Cout=32, Rout=6, 5, H, W]
    data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W]
);

// 4. 辅助函数：从 weight 和 kernel_expansion_idx 构造完整卷积核
void get_kernel(
    const data_t weight[COUT][CIN][RIN][7],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    data_t kernel[COUT][ROUT][CIN][RIN][KERNEL_H][KERNEL_W]
);

// 5. 标准 2D 卷积（在重排后的数据上执行）
void conv2d_3x3(
    data_t input[(CIN*RIN)][(CHARTS*H_PADDED)][W_PADDED],
    const data_t kernel[(COUT*ROUT)][(CIN*RIN)][KERNEL_H][KERNEL_W],
    const data_t bias[COUT*ROUT],
    data_t output[(COUT*ROUT)][(CHARTS*H_PADDED)][W_PADDED]
);

#endif // ICO_CONV_LAYER0_HPP
