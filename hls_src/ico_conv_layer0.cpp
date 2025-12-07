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

// ==================== 2. PadIco 实现 ====================
void pad_ico(
    data_t input[RIN][CHARTS][H][W],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[RIN][CHARTS][H_PADDED][W_PADDED]
) {
    // 先 flatten 输入为 1D 数组（与 PyTorch einops.rearrange 对应）
    const int flat_size = RIN * CHARTS * H * W;
    data_t input_flat[flat_size];
    
    int idx = 0;
    for (int r = 0; r < RIN; r++) {
        for (int c = 0; c < CHARTS; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    input_flat[idx++] = input[r][c][h][w];
                }
            }
        }
    }
    
    // 通过 reorder_idx 查表重排
    for (int r = 0; r < RIN; r++) {
        for (int c = 0; c < CHARTS; c++) {
            for (int h = 0; h < H_PADDED; h++) {
                for (int w = 0; w < W_PADDED; w++) {
                    int src_idx = reorder_idx[r][c][h][w];
                    if (src_idx >= 0 && src_idx < flat_size) {
                        output[r][c][h][w] = input_flat[src_idx];
                    } else {
                        output[r][c][h][w] = 0.0f;  // 越界填 0
                    }
                }
            }
        }
    }
}

// ==================== 3. get_kernel 实现 ====================
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

// ==================== 4. conv2d_3x3 标准 2D 卷积 ====================
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

// ==================== 5. ConvIco Layer0 主函数 ====================
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
        
        // 2.1 CleanVertices + PadIco
        data_t input_frame[RIN][CHARTS][H][W];
        data_t padded_frame[RIN][CHARTS][H_PADDED][W_PADDED];
        
        // 提取当前帧
        for (int ci = 0; ci < CIN; ci++) {
            for (int ri = 0; ri < RIN; ri++) {
                for (int c = 0; c < CHARTS; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            input_frame[ri][c][h][w] = input[t][ci][ri][c][h][w];
                        }
                    }
                }
            }
        }
        
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
        
        // 2.5 CleanVertices
        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                for (int c = 0; c < CHARTS; c++) {
                    output[t][co][ro][c][0][0] = 0.0f;
                    output[t][co][ro][c][0][H] = 0.0f;
                }
            }
        }
    }
}
