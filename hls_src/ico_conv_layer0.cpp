#include "ico_conv_layer0.hpp"
#include <cstring>

/*
实现说明
========
- 本文件以硬件友好的方式实现 IcoCNN 的 layer0。
- 当前顶层路径移除了两个较大的中间缓冲：
  1) conv_output
  2) output_frame
  以降低 BRAM 和控制逻辑开销。
- 极点平滑在 output[t] 上原地执行。
*/

// 1) 清理极点顶点
void clean_vertices(
    data_t input[CHARTS][H][W],
    data_t output[CHARTS][H][W]
) {
    // 步骤 1：拷贝全部元素。
    for (int c = 0; c < CHARTS; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = input[c][h][w];
            }
        }
    }
    
    // 步骤 2：清零每个图表上的两个极点位置。
    for (int c = 0; c < CHARTS; c++) {
        output[c][0][0] = 0.0f;
        output[c][0][H] = 0.0f;
    }
}

// 2) 极点平滑
void smooth_vertices(
    data_t input[CIN][RIN][CHARTS][H][W],
    data_t output[CIN][RIN][CHARTS][H][W]
) {
    // 步骤 1：先进行一份完全拷贝。
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        output[ci][ri][c][h][w] = input[ci][ri][c][h][w];
                    }
                }
            }
        }
    }
    
    // 步骤 2：先清零极点顶点，随后用均值替换。
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                output[ci][ri][c][0][0] = 0.0f;
                output[ci][ri][c][0][H] = 0.0f;
            }
        }
    }
    
    // 步骤 3：根据各自的 5 邻域计算两个极点值。
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                int prev_c = (c - 1 + CHARTS) % CHARTS;
                
                float sum_v1 = 0.0f;
                sum_v1 += input[ci][ri][c][1][0];        // 邻点坐标：(c, 1, 0)
                sum_v1 += input[ci][ri][c][1][1];        // 邻点坐标：(c, 1, 1)
                sum_v1 += input[ci][ri][c][0][1];        // 邻点坐标：(c, 0, 1)
                sum_v1 += input[ci][ri][prev_c][H-1][H]; // 邻点坐标：(c-1, H-1, H)
                sum_v1 += input[ci][ri][prev_c][H-1][H-1]; // 邻点坐标：(c-1, H-1, H-1)
                output[ci][ri][c][0][0] = sum_v1 / 5.0f;
                
                float sum_v2 = 0.0f;
                sum_v2 += input[ci][ri][c][1][H];        // 邻点坐标：(c, 1, H)
                sum_v2 += input[ci][ri][c][1][(H+1)%W]; // 邻点坐标：(c, 1, H+1)
                sum_v2 += input[ci][ri][c][0][(H+1)%W]; // 邻点坐标：(c, 0, H+1)
                sum_v2 += input[ci][ri][prev_c][H-1][W-1]; // 邻点坐标：(c-1, H-1, W-1)
                sum_v2 += input[ci][ri][c][0][H-1];     // 邻点坐标：(c, 0, H-1)
                output[ci][ri][c][0][H] = sum_v2 / 5.0f;
            }
        }
    }
}

// 3) PadIco 填充与重排
void pad_ico(
    data_t input[CIN][RIN][CHARTS][H][W],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[RIN][CHARTS][H_PADDED][W_PADDED]
) {
    // 阶段 A：先对输入极点做平滑。
    static data_t input_after_smooth[CIN][RIN][CHARTS][H][W];
    smooth_vertices(input, input_after_smooth);
    
    // 阶段 B：计算填充边界需要的全局极点值。
    float smooth_north_pole_sum = 0.0f;
    float smooth_south_pole_sum = 0.0f;
    
    for (int ri = 0; ri < RIN; ri++) {
        for (int c = 0; c < CHARTS; c++) {
            smooth_north_pole_sum += input_after_smooth[0][ri][c][H-1][0];
            smooth_south_pole_sum += input_after_smooth[0][ri][c][0][W-1];
        }
    }
    float smooth_north_pole = smooth_north_pole_sum / (RIN * CHARTS);
    float smooth_south_pole = smooth_south_pole_sum / (RIN * CHARTS);
    
    // 阶段 C：按索引表重排填充张量。
    for (int ri = 0; ri < RIN; ri++) {
        for (int c = 0; c < CHARTS; c++) {
            for (int h = 0; h < H_PADDED; h++) {
                for (int w = 0; w < W_PADDED; w++) {
                    int reorder_val = reorder_idx[ri][c][h][w];
                    
                    int src_chart = reorder_val / (H * W);
                    int remainder = reorder_val % (H * W);
                    int src_h = remainder / W;
                    int src_w = remainder % W;
                    
                    output[ri][c][h][w] = input_after_smooth[0][ri][src_chart][src_h][src_w];
                }
            }
        }
    }
    
    // 阶段 D：在固定填充位置写入显式极点值。
    for (int ri = 0; ri < RIN; ri++) {
        for (int c = 0; c < CHARTS; c++) {
            output[ri][c][H_PADDED-1][1] = smooth_north_pole;
            output[ri][c][1][W_PADDED-1] = smooth_south_pole;
        }
    }
}

// 4) 构建展开后的卷积核
void get_kernel(
    const data_t weight[COUT][CIN][RIN][7],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    data_t kernel[COUT][ROUT][CIN][RIN][KERNEL_H][KERNEL_W]
) {
    // 步骤 1：清零完整 3x3 卷积核。
    for (int co = 0; co < COUT; co++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int ci = 0; ci < CIN; ci++) {
                for (int ri = 0; ri < RIN; ri++) {
                    for (int kh = 0; kh < KERNEL_H; kh++) {
#pragma HLS UNROLL
                        for (int kw = 0; kw < KERNEL_W; kw++) {
#pragma HLS UNROLL
                            kernel[co][ro][ci][ri][kh][kw] = 0.0f;
                        }
                    }
                }
            }
        }
    }
    
    // 步骤 2：将紧凑 7 邻域权重放到展开后的 3x3 位置。
    for (int co = 0; co < COUT; co++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int ci = 0; ci < CIN; ci++) {
                for (int ri = 0; ri < RIN; ri++) {
                    for (int k = 0; k < 9; k++) {
                        int idx_cout = kernel_expansion_idx[co][ro][ci][ri][k][0];
                        int idx_cin  = kernel_expansion_idx[co][ro][ci][ri][k][1];
                        int idx_rin  = kernel_expansion_idx[co][ro][ci][ri][k][2];
                        int idx_w    = kernel_expansion_idx[co][ro][ci][ri][k][3];
                        
                        int kh = k / 3;
                        int kw = k % 3;
                        
                        if (idx_w >= 0 && idx_w < 7) {
                            kernel[co][ro][ci][ri][kh][kw] = 
                                weight[idx_cout][idx_cin][idx_rin][idx_w];
                        }
                    }
                }
            }
        }
    }
    
    // 步骤 3：将结构上无效的两个位置强制置零。
    for (int co = 0; co < COUT; co++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int ci = 0; ci < CIN; ci++) {
                for (int ri = 0; ri < RIN; ri++) {
                    kernel[co][ro][ci][ri][0][2] = 0.0f;
                    kernel[co][ro][ci][ri][2][0] = 0.0f;
                }
            }
        }
    }
}

// 5) 标准二维卷积辅助函数
void conv2d_3x3(
    data_t input[(CIN*RIN)][(CHARTS*H_PADDED)][W_PADDED],
    const data_t kernel[(COUT*ROUT)][(CIN*RIN)][KERNEL_H][KERNEL_W],
    const data_t bias[COUT*ROUT],
    data_t output[(COUT*ROUT)][(CHARTS*H_PADDED)][W_PADDED]
) {
    // 保持为独立模块，便于控制 HLS 调度。
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=kernel cyclic factor=OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OC_PAR_FACTOR dim=1

    const int IN_CH = CIN * RIN;
    const int OUT_CH = COUT * ROUT;
    const int IN_H = CHARTS * H_PADDED;
    const int IN_W = W_PADDED;
    
    // 在填充网格上执行完整卷积。
    for (int oc = 0; oc < OUT_CH; oc++) {
#pragma HLS UNROLL factor=OC_PAR_FACTOR
        for (int oh = 0; oh < IN_H; oh++) {
            for (int ow = 0; ow < IN_W; ow++) {
#pragma HLS PIPELINE II=1
                data_t sum = bias[oc];
                
                for (int ic = 0; ic < IN_CH; ic++) {
                    for (int kh = 0; kh < KERNEL_H; kh++) {
                        for (int kw = 0; kw < KERNEL_W; kw++) {
                            int ih = oh + kh - 1;
                            int iw = ow + kw - 1;
                            
                            if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                sum += input[ic][ih][iw] * kernel[oc][ic][kh][kw];
                            }
                        }
                    }
                }
                
                output[oc][oh][ow] = sum;
            }
        }
    }
}

// 6) 顶层函数：layer0 ConvIco
void conv_ico_layer0(
    data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W],
    const data_t weight[COUT][CIN][RIN][7],
    const data_t bias[COUT],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W]
) {
    // 控制顶层调度，并设置适中的输出通道并行度。
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=OC_PAR_FACTOR dim=1

    // 预计算一次展开卷积核，供所有时间步共享。
    static data_t kernel[COUT][ROUT][CIN][RIN][KERNEL_H][KERNEL_W];
#pragma HLS ARRAY_PARTITION variable=kernel cyclic factor=OC_PAR_FACTOR dim=1
    get_kernel(weight, kernel_expansion_idx, kernel);

    // 主循环：逐帧处理以降低内存占用。
    for (int t = 0; t < TIME_STEPS; t++) {
        // 阶段 1：提取输入第 t 帧。
        data_t input_frame[CIN][RIN][CHARTS][H][W];
        data_t padded_frame[RIN][CHARTS][H_PADDED][W_PADDED];
        for (int ci = 0; ci < CIN; ci++) {
            for (int ri = 0; ri < RIN; ri++) {
                for (int c = 0; c < CHARTS; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            input_frame[ci][ri][c][h][w] = input[t][ci][ri][c][h][w];
                        }
                    }
                }
            }
        }

        // 阶段 2：平滑 + 填充/重排到图表填充布局。
        pad_ico(input_frame, reorder_idx, padded_frame);

        // 阶段 3：重塑 [RIN][CHARTS][H_PADDED][W_PADDED] -> [IN_CH][IN_H][IN_W]。
        data_t reshaped_input[CIN * RIN][CHARTS * H_PADDED][W_PADDED];
        int ch_idx = 0;
        for (int ci = 0; ci < CIN; ci++) {
            for (int ri = 0; ri < RIN; ri++) {
                for (int c = 0; c < CHARTS; c++) {
                    for (int h = 0; h < H_PADDED; h++) {
                        for (int w = 0; w < W_PADDED; w++) {
                            reshaped_input[ch_idx][c * H_PADDED + h][w] = padded_frame[ri][c][h][w];
                        }
                    }
                }
                ch_idx++;
            }
        }

        // 阶段 4：拉平 kernel 与 bias，用于直接循环卷积。
        data_t kernel_2d[COUT * ROUT][CIN * RIN][KERNEL_H][KERNEL_W];
        data_t bias_2d[COUT * ROUT];
#pragma HLS ARRAY_PARTITION variable=kernel_2d cyclic factor=OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=bias_2d cyclic factor=OC_PAR_FACTOR dim=1
        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                int out_idx = co * ROUT + ro;
                bias_2d[out_idx] = bias[co];
                for (int ci = 0; ci < CIN; ci++) {
                    for (int ri = 0; ri < RIN; ri++) {
                        int in_idx = ci * RIN + ri;
                        for (int kh = 0; kh < KERNEL_H; kh++) {
                            for (int kw = 0; kw < KERNEL_W; kw++) {
                                kernel_2d[out_idx][in_idx][kh][kw] = kernel[co][ro][ci][ri][kh][kw];
                            }
                        }
                    }
                }
            }
        }

        // 阶段 5：仅计算有效输出区域并直接写入 output[t]。
        // 这一步替代了旧的 conv_output 中间缓冲。
        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                int out_idx = co * ROUT + ro;
                for (int c = 0; c < CHARTS; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            int oh = c * H_PADDED + h + 1;
                            int ow = w + 1;
                            data_t sum = bias_2d[out_idx];
                            for (int ic = 0; ic < CIN * RIN; ic++) {
                                for (int kh = 0; kh < KERNEL_H; kh++) {
                                    for (int kw = 0; kw < KERNEL_W; kw++) {
                                        int ih = oh + kh - 1;
                                        int iw = ow + kw - 1;
                                        sum += reshaped_input[ic][ih][iw] * kernel_2d[out_idx][ic][kh][kw];
                                    }
                                }
                            }
                            output[t][co][ro][c][h][w] = sum;
                        }
                    }
                }
            }
        }

        // 阶段 6：在 output[t] 上原地做极点平滑。
        // 这一步替代了旧的 output_frame 中间缓冲。
        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                for (int c = 0; c < CHARTS; c++) {
                    output[t][co][ro][c][0][0] = 0.0f;
                    output[t][co][ro][c][0][H] = 0.0f;
                }
            }
        }
        for (int co = 0; co < COUT; co++) {
            for (int c = 0; c < CHARTS; c++) {
                int prev_c = (c - 1 + CHARTS) % CHARTS;
                float sum_v1 = 0.0f;
                float sum_v2 = 0.0f;
                for (int ro = 0; ro < ROUT; ro++) {
                    sum_v1 += output[t][co][ro][c][1][0];
                    sum_v1 += output[t][co][ro][c][1][1];
                    sum_v1 += output[t][co][ro][c][0][1];
                    sum_v1 += output[t][co][ro][prev_c][H - 1][H];
                    sum_v1 += output[t][co][ro][prev_c][H - 1][H - 1];
                    sum_v2 += output[t][co][ro][c][1][H];
                    sum_v2 += output[t][co][ro][c][1][(H + 1) % W];
                    sum_v2 += output[t][co][ro][c][0][(H + 1) % W];
                    sum_v2 += output[t][co][ro][prev_c][H - 1][W - 1];
                    sum_v2 += output[t][co][ro][c][0][H - 1];
                }
                float mean_v1 = sum_v1 / (ROUT * 5.0f);
                float mean_v2 = sum_v2 / (ROUT * 5.0f);
                for (int ro = 0; ro < ROUT; ro++) {
                    output[t][co][ro][c][0][0] = mean_v1;
                    output[t][co][ro][c][0][H] = mean_v2;
                }
            }
        }
    }
}
