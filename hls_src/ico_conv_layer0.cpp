#include "ico_conv_layer0.hpp"
#include <cstring>

// ==================== 1. CleanVertices 实现 ====================
void clean_vertices(
    data_t input[CHARTS][H][W],
    data_t output[CHARTS][H][W]
) {
    // 复制数据
    for (int c = 0; c < CHARTS; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = input[c][h][w];
            }
        }
    }
    
    // 清零顶点: [0,0] 和 [0, 2^r] = [0, 4]
    for (int c = 0; c < CHARTS; c++) {
        output[c][0][0] = 0.0f;
        output[c][0][H] = 0.0f;  // W = 2*H, 所以 2^r = H
    }
}

// ==================== 2. SmoothVertices 实现 ====================
void smooth_vertices(
    data_t input[CIN][RIN][CHARTS][H][W],
    data_t output[CIN][RIN][CHARTS][H][W]
) {
    // 首先复制输入到输出
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
    
    // CleanVertices: 将顶点位置清零
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                output[ci][ri][c][0][0] = 0.0f;  // v1 北极
                output[ci][ri][c][0][H] = 0.0f;  // v2 南极 (W/2 = H = 4)
            }
        }
    }
    
    // SmoothVertices: 用邻居均值替换顶点
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                int prev_c = (c - 1 + CHARTS) % CHARTS;
                
                // v1 (0,0) 的 5 个邻居的均值
                float sum_v1 = 0.0f;
                sum_v1 += input[ci][ri][c][1][0];        // (c, 1, 0)
                sum_v1 += input[ci][ri][c][1][1];        // (c, 1, 1)
                sum_v1 += input[ci][ri][c][0][1];        // (c, 0, 1)
                sum_v1 += input[ci][ri][prev_c][H-1][H]; // (c-1, H-1, H)
                sum_v1 += input[ci][ri][prev_c][H-1][H-1]; // (c-1, H-1, H-1)
                output[ci][ri][c][0][0] = sum_v1 / 5.0f;
                
                // v2 (0,H) 的 5 个邻居的均值
                float sum_v2 = 0.0f;
                sum_v2 += input[ci][ri][c][1][H];        // (c, 1, H)
                sum_v2 += input[ci][ri][c][1][(H+1)%W]; // (c, 1, H+1)
                sum_v2 += input[ci][ri][c][0][(H+1)%W]; // (c, 0, H+1)
                sum_v2 += input[ci][ri][prev_c][H-1][W-1]; // (c-1, H-1, W-1)
                sum_v2 += input[ci][ri][c][0][H-1];     // (c, 0, H-1)
                output[ci][ri][c][0][H] = sum_v2 / 5.0f;
            }
        }
    }
}

// ==================== 3. PadIco 实现（含极点平滑） ====================
void pad_ico(
    data_t input[CIN][RIN][CHARTS][H][W],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[RIN][CHARTS][H_PADDED][W_PADDED]
) {
    // 步骤 1: 先对输入应用 SmoothVertices
    static data_t input_after_smooth[CIN][RIN][CHARTS][H][W];
    smooth_vertices(input, input_after_smooth);
    
    // 步骤 2: 计算极点平滑值（所有 charts 上的极点平均）
    float smooth_north_pole_sum = 0.0f;
    float smooth_south_pole_sum = 0.0f;
    
    for (int ri = 0; ri < RIN; ri++) {
        for (int c = 0; c < CHARTS; c++) {
            smooth_north_pole_sum += input_after_smooth[0][ri][c][H-1][0];   // north pole 位置
            smooth_south_pole_sum += input_after_smooth[0][ri][c][0][W-1];   // south pole 位置
        }
    }
    float smooth_north_pole = smooth_north_pole_sum / (RIN * CHARTS);
    float smooth_south_pole = smooth_south_pole_sum / (RIN * CHARTS);
    
    // 步骤 3: 执行 PadIco（使用 reorder_idx 重排）
    for (int ri = 0; ri < RIN; ri++) {
        for (int c = 0; c < CHARTS; c++) {
            for (int h = 0; h < H_PADDED; h++) {
                for (int w = 0; w < W_PADDED; w++) {
                    int reorder_val = reorder_idx[ri][c][h][w];
                    
                    // 计算源位置
                    int src_chart = reorder_val / (H * W);
                    int remainder = reorder_val % (H * W);
                    int src_h = remainder / W;
                    int src_w = remainder % W;
                    
                    // 从 smooth 后的数据中读取
                    output[ri][c][h][w] = input_after_smooth[0][ri][src_chart][src_h][src_w];
                }
            }
        }
    }
    
    // 步骤 4: 设置极点平滑值
    for (int ri = 0; ri < RIN; ri++) {
        for (int c = 0; c < CHARTS; c++) {
            output[ri][c][H_PADDED-1][1] = smooth_north_pole;  // 北极位置
            output[ri][c][1][W_PADDED-1] = smooth_south_pole;  // 南极位置
        }
    }
}

// ==================== 4. get_kernel 实现 ====================
void get_kernel(
    const data_t weight[COUT][CIN][RIN][7],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    data_t kernel[COUT][ROUT][CIN][RIN][KERNEL_H][KERNEL_W]
) {
    // 初始化为 0
    for (int co = 0; co < COUT; co++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int ci = 0; ci < CIN; ci++) {
                for (int ri = 0; ri < RIN; ri++) {
                    for (int kh = 0; kh < KERNEL_H; kh++) {
                        for (int kw = 0; kw < KERNEL_W; kw++) {
                            kernel[co][ro][ci][ri][kh][kw] = 0.0f;
                        }
                    }
                }
            }
        }
    }
    
    // 通过 kernel_expansion_idx 从 weight[Cout,Cin,Rin,7] 展开到 3x3 kernel
    for (int co = 0; co < COUT; co++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int ci = 0; ci < CIN; ci++) {
                for (int ri = 0; ri < RIN; ri++) {
                    for (int k = 0; k < 9; k++) {  // 9 个位置 (3x3)
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
    
    // 清零特定位置: kernel[..., 0, 2] = 0 和 kernel[..., 2, 0] = 0
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

// ==================== 5. conv2d_3x3 标准 2D 卷积 ====================
void conv2d_3x3(
    data_t input[(CIN*RIN)][(CHARTS*H_PADDED)][W_PADDED],
    const data_t kernel[(COUT*ROUT)][(CIN*RIN)][KERNEL_H][KERNEL_W],
    const data_t bias[COUT*ROUT],
    data_t output[(COUT*ROUT)][(CHARTS*H_PADDED)][W_PADDED]
) {
    const int IN_CH = CIN * RIN;
    const int OUT_CH = COUT * ROUT;
    const int IN_H = CHARTS * H_PADDED;
    const int IN_W = W_PADDED;
    
    // 卷积计算
    for (int oc = 0; oc < OUT_CH; oc++) {
        for (int oh = 0; oh < IN_H; oh++) {
            for (int ow = 0; ow < IN_W; ow++) {
                data_t sum = bias[oc];
                
                for (int ic = 0; ic < IN_CH; ic++) {
                    for (int kh = 0; kh < KERNEL_H; kh++) {
                        for (int kw = 0; kw < KERNEL_W; kw++) {
                            int ih = oh + kh - 1;  // padding=1 的效果
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

// ==================== 6. ConvIco Layer0 主函数 ====================
void conv_ico_layer0(
    data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W],
    const data_t weight[COUT][CIN][RIN][7],
    const data_t bias[COUT],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W]
) {
    // 1. 预计算卷积核（只需计算一次）
    static data_t kernel[COUT][ROUT][CIN][RIN][KERNEL_H][KERNEL_W];
    
    get_kernel(weight, kernel_expansion_idx, kernel);
    
    // 2. 逐帧处理
    for (int t = 0; t < TIME_STEPS; t++) {
        
        // 2.1 提取当前帧并执行 SmoothVertices + PadIco
        data_t input_frame[CIN][RIN][CHARTS][H][W];
        data_t padded_frame[RIN][CHARTS][H_PADDED][W_PADDED];
        
        // 提取当前帧
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
        
        // PadIco (内部包含 SmoothVertices)
        pad_ico(input_frame, reorder_idx, padded_frame);
        
        // 2.2 Reshape 为 2D 卷积格式
        data_t reshaped_input[CIN*RIN][CHARTS*H_PADDED][W_PADDED];
        
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
        
        // 2.3 标准 2D 卷积
        data_t conv_output[COUT*ROUT][CHARTS*H_PADDED][W_PADDED];
        
        // Flatten kernel
        data_t kernel_2d[COUT*ROUT][CIN*RIN][KERNEL_H][KERNEL_W];
        data_t bias_2d[COUT*ROUT];
        
        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                int out_idx = co * ROUT + ro;
                bias_2d[out_idx] = bias[co];
                
                for (int ci = 0; ci < CIN; ci++) {
                    for (int ri = 0; ri < RIN; ri++) {
                        int in_idx = ci * RIN + ri;
                        for (int kh = 0; kh < KERNEL_H; kh++) {
                            for (int kw = 0; kw < KERNEL_W; kw++) {
                                kernel_2d[out_idx][in_idx][kh][kw] = 
                                    kernel[co][ro][ci][ri][kh][kw];
                            }
                        }
                    }
                }
            }
        }
        
        conv2d_3x3(reshaped_input, kernel_2d, bias_2d, conv_output);
        
        // 2.4 Reshape 回 icosahedral 格式并去掉 padding
        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                int out_idx = co * ROUT + ro;
                for (int c = 0; c < CHARTS; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            // 去掉 padding: [1:-1, 1:-1]
                            output[t][co][ro][c][h][w] = 
                                conv_output[out_idx][c * H_PADDED + h + 1][w + 1];
                        }
                    }
                }
            }
        }
        
        // 2.5 对输出应用 SmoothVertices（与 Python 的 forward 最后一步对应）
        data_t output_frame[COUT][ROUT][CHARTS][H][W];
        
        // 提取当前帧输出
        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                for (int c = 0; c < CHARTS; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            output_frame[co][ro][c][h][w] = output[t][co][ro][c][h][w];
                        }
                    }
                }
            }
        }
        
        // CleanVertices: 清零顶点
        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                for (int c = 0; c < CHARTS; c++) {
                    output_frame[co][ro][c][0][0] = 0.0f;  // v1
                    output_frame[co][ro][c][0][H] = 0.0f;  // v2
                }
            }
        }
        
        // SmoothVertices: 用邻居均值替换顶点
        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                for (int c = 0; c < CHARTS; c++) {
                    int prev_c = (c - 1 + CHARTS) % CHARTS;
                    
                    // v1 (0,0) 的 5 个邻居的均值
                    float sum_v1 = 0.0f;
                    sum_v1 += output[t][co][ro][c][1][0];
                    sum_v1 += output[t][co][ro][c][1][1];
                    sum_v1 += output[t][co][ro][c][0][1];
                    sum_v1 += output[t][co][ro][prev_c][H-1][H];
                    sum_v1 += output[t][co][ro][prev_c][H-1][H-1];
                    output_frame[co][ro][c][0][0] = sum_v1 / 5.0f;
                    
                    // v2 (0,H) 的 5 个邻居的均值
                    float sum_v2 = 0.0f;
                    sum_v2 += output[t][co][ro][c][1][H];
                    sum_v2 += output[t][co][ro][c][1][(H+1)%W];
                    sum_v2 += output[t][co][ro][c][0][(H+1)%W];
                    sum_v2 += output[t][co][ro][prev_c][H-1][W-1];
                    sum_v2 += output[t][co][ro][c][0][H-1];
                    output_frame[co][ro][c][0][H] = sum_v2 / 5.0f;
                }
            }
        }
        
        // 写回输出
        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                for (int c = 0; c < CHARTS; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            output[t][co][ro][c][h][w] = output_frame[co][ro][c][h][w];
                        }
                    }
                }
            }
        }
    }
}
