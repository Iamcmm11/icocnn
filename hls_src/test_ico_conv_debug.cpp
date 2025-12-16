/**
 * Layer0 中间层调试程序 - C++ 端
 * 
 * 输出 ConvIco Layer0 的所有中间计算结果（只针对第0帧）
 * 输出文件保存到: ../hls_testdata/layer0/debug_intermediate_cpp/
 * 
 * 对应 Python 端的 debug_layer0_intermediate.py
 */

#include "ico_conv_layer0.hpp"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

// 创建输出目录
const std::string DEBUG_DIR = "../hls_testdata/layer0/debug_intermediate_cpp/";

/**
 * 保存 2D 矩阵到文本文件（MATLAB 风格）
 */
template<int HEIGHT, int WIDTH>
void save_matrix_2d(const std::string& filename, data_t arr[HEIGHT][WIDTH], const std::string& name) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot create " << filename << std::endl;
        return;
    }
    
    f << "# " << name << std::endl;
    f << "# Shape: (" << HEIGHT << ", " << WIDTH << ")" << std::endl;
    
    // 计算统计量
    double min_val = arr[0][0], max_val = arr[0][0], sum = 0.0;
    for (int h = 0; h < HEIGHT; h++) {
        for (int w = 0; w < WIDTH; w++) {
            if (arr[h][w] < min_val) min_val = arr[h][w];
            if (arr[h][w] > max_val) max_val = arr[h][w];
            sum += arr[h][w];
        }
    }
    double mean = sum / (HEIGHT * WIDTH);
    
    f << "# Min: " << std::fixed << std::setprecision(8) << min_val 
      << ", Max: " << max_val << ", Mean: " << mean << std::endl;
    f << "#" << std::string(70, '=') << std::endl << std::endl;
    
    // 输出矩阵
    f << std::fixed << std::setprecision(6);
    for (int h = 0; h < HEIGHT; h++) {
        f << "  ";
        for (int w = 0; w < WIDTH; w++) {
            f << std::setw(10) << arr[h][w] << "  ";
        }
        f << std::endl;
    }
    
    f.close();
    std::cout << "  Saved: " << filename << std::endl;
}

/**
 * 保存 3D 张量到文本文件（逐层输出）
 */
template<int CHANNELS, int HEIGHT, int WIDTH>
void save_tensor_3d(const std::string& filename, data_t arr[CHANNELS][HEIGHT][WIDTH], const std::string& name) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot create " << filename << std::endl;
        return;
    }
    
    f << "# " << name << std::endl;
    f << "# Shape: (" << CHANNELS << ", " << HEIGHT << ", " << WIDTH << ")" << std::endl;
    
    // 计算统计量
    double min_val = arr[0][0][0], max_val = arr[0][0][0], sum = 0.0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int h = 0; h < HEIGHT; h++) {
            for (int w = 0; w < WIDTH; w++) {
                if (arr[c][h][w] < min_val) min_val = arr[c][h][w];
                if (arr[c][h][w] > max_val) max_val = arr[c][h][w];
                sum += arr[c][h][w];
            }
        }
    }
    double mean = sum / (CHANNELS * HEIGHT * WIDTH);
    
    f << "# Min: " << std::fixed << std::setprecision(8) << min_val 
      << ", Max: " << max_val << ", Mean: " << mean << std::endl;
    f << "#" << std::string(70, '=') << std::endl << std::endl;
    
    // 输出每个通道（只输出前5个通道，避免文件过大）
    f << std::fixed << std::setprecision(6);
    int max_channels = (CHANNELS > 5) ? 5 : CHANNELS;
    
    for (int c = 0; c < max_channels; c++) {
        f << "# [Channel " << c << "] - Shape: (" << HEIGHT << ", " << WIDTH << ")" << std::endl;
        for (int h = 0; h < HEIGHT; h++) {
            f << "  ";
            for (int w = 0; w < WIDTH; w++) {
                f << std::setw(10) << arr[c][h][w] << "  ";
            }
            f << std::endl;
        }
        f << std::endl;
    }
    
    if (CHANNELS > 5) {
        f << "# ... (省略其余 " << (CHANNELS - 5) << " 个通道)" << std::endl;
    }
    
    f.close();
    std::cout << "  Saved: " << filename << std::endl;
}

/**
 * 保存 4D 张量（icosahedral 格式）
 */
template<int R_DIM, int CHARTS_DIM, int H_DIM, int W_DIM>
void save_ico_tensor_4d(const std::string& filename, data_t arr[R_DIM][CHARTS_DIM][H_DIM][W_DIM], const std::string& name) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot create " << filename << std::endl;
        return;
    }
    
    f << "# " << name << std::endl;
    f << "# Shape: (" << R_DIM << ", " << CHARTS_DIM << ", " << H_DIM << ", " << W_DIM << ")" << std::endl;
    
    // 计算统计量
    double min_val = arr[0][0][0][0], max_val = arr[0][0][0][0], sum = 0.0;
    for (int r = 0; r < R_DIM; r++) {
        for (int ch = 0; ch < CHARTS_DIM; ch++) {
            for (int h = 0; h < H_DIM; h++) {
                for (int w = 0; w < W_DIM; w++) {
                    if (arr[r][ch][h][w] < min_val) min_val = arr[r][ch][h][w];
                    if (arr[r][ch][h][w] > max_val) max_val = arr[r][ch][h][w];
                    sum += arr[r][ch][h][w];
                }
            }
        }
    }
    double mean = sum / (R_DIM * CHARTS_DIM * H_DIM * W_DIM);
    
    f << "# Min: " << std::fixed << std::setprecision(8) << min_val 
      << ", Max: " << max_val << ", Mean: " << mean << std::endl;
    f << "#" << std::string(70, '=') << std::endl << std::endl;
    
    // 输出每个 chart
    f << std::fixed << std::setprecision(6);
    for (int r = 0; r < R_DIM; r++) {
        for (int ch = 0; ch < CHARTS_DIM; ch++) {
            f << "# [R" << r << ", chart" << ch << "] - Shape: (" << H_DIM << ", " << W_DIM << ")" << std::endl;
            for (int h = 0; h < H_DIM; h++) {
                f << "  ";
                for (int w = 0; w < W_DIM; w++) {
                    f << std::setw(10) << arr[r][ch][h][w] << "  ";
                }
                f << std::endl;
            }
            f << std::endl;
        }
    }
    
    f.close();
    std::cout << "  Saved: " << filename << std::endl;
}

/**
 * 保存 5D 输出张量（只输出前3个通道）
 */
template<int CO_DIM, int RO_DIM, int CH_DIM, int H_DIM, int W_DIM>
void save_output_5d(const std::string& filename, data_t arr[CO_DIM][RO_DIM][CH_DIM][H_DIM][W_DIM], const std::string& name) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot create " << filename << std::endl;
        return;
    }
    
    f << "# " << name << std::endl;
    f << "# Shape: (" << CO_DIM << ", " << RO_DIM << ", " << CH_DIM << ", " << H_DIM << ", " << W_DIM << ")" << std::endl;
    
    // 计算统计量
    double min_val = arr[0][0][0][0][0], max_val = arr[0][0][0][0][0], sum = 0.0;
    int count = 0;
    for (int co = 0; co < CO_DIM; co++) {
        for (int ro = 0; ro < RO_DIM; ro++) {
            for (int ch = 0; ch < CH_DIM; ch++) {
                for (int h = 0; h < H_DIM; h++) {
                    for (int w = 0; w < W_DIM; w++) {
                        if (arr[co][ro][ch][h][w] < min_val) min_val = arr[co][ro][ch][h][w];
                        if (arr[co][ro][ch][h][w] > max_val) max_val = arr[co][ro][ch][h][w];
                        sum += arr[co][ro][ch][h][w];
                        count++;
                    }
                }
            }
        }
    }
    double mean = sum / count;
    
    f << "# Min: " << std::fixed << std::setprecision(8) << min_val 
      << ", Max: " << max_val << ", Mean: " << mean << std::endl;
    f << "#" << std::string(70, '=') << std::endl << std::endl;
    
    // 只输出前3个输出通道
    f << std::fixed << std::setprecision(6);
    int max_co = (CO_DIM > 3) ? 3 : CO_DIM;
    
    for (int co = 0; co < max_co; co++) {
        for (int ro = 0; ro < RO_DIM; ro++) {
            for (int ch = 0; ch < CH_DIM; ch++) {
                f << "# [C" << co << ", R" << ro << ", chart" << ch << "] - Shape: (" << H_DIM << ", " << W_DIM << ")" << std::endl;
                for (int h = 0; h < H_DIM; h++) {
                    f << "  ";
                    for (int w = 0; w < W_DIM; w++) {
                        f << std::setw(10) << arr[co][ro][ch][h][w] << "  ";
                    }
                    f << std::endl;
                }
                f << std::endl;
            }
        }
    }
    
    if (CO_DIM > 3) {
        f << "# ... (省略其余 " << (CO_DIM - 3) << " 个通道)" << std::endl;
    }
    
    f.close();
    std::cout << "  Saved: " << filename << std::endl;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "Layer0 中间层调试 - C++ 端" << std::endl;
    std::cout << "======================================================================" << std::endl;
    
    const std::string data_dir = "../hls_testdata/layer0/";
    
    // ==================== 1. 读取数据 ====================
    std::cout << "\n[1] Loading data..." << std::endl;
    
    auto input_vec = read_txt_data(data_dir + "input_rearranged.txt");
    auto weight_vec = read_txt_data(data_dir + "weight.txt");
    auto bias_vec = read_txt_data(data_dir + "bias.txt");
    auto kernel_exp_idx_vec = read_txt_data_int(data_dir + "kernel_expansion_idx.txt");
    auto reorder_idx_vec = read_txt_data_int(data_dir + "reorder_idx.txt");
    
    std::cout << "  Input shape: (" << TIME_STEPS << ", " << CIN << ", " << RIN << ", " 
              << CHARTS << ", " << H << ", " << W << ")" << std::endl;
    std::cout << "  Weight shape: (" << COUT << ", " << CIN << ", " << RIN << ", 7)" << std::endl;
    
    // ==================== 2. 分配数组 ====================
    std::cout << "\n[2] Preparing arrays..." << std::endl;
    
    static data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W];
    static data_t weight[COUT][CIN][RIN][7];
    static data_t bias[COUT];
    static int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4];
    static int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED];
    
    // 填充数据（与原 test_ico_conv.cpp 相同的逻辑）
    int idx = 0;
    for (int t = 0; t < TIME_STEPS; t++)
        for (int ci = 0; ci < CIN; ci++)
            for (int ri = 0; ri < RIN; ri++)
                for (int c = 0; c < CHARTS; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            input[t][ci][ri][c][h][w] = input_vec[idx++];
    
    idx = 0;
    for (int co = 0; co < COUT; co++)
        for (int ci = 0; ci < CIN; ci++)
            for (int ri = 0; ri < RIN; ri++)
                for (int k = 0; k < 7; k++)
                    weight[co][ci][ri][k] = weight_vec[idx++];
    
    for (int co = 0; co < COUT; co++)
        bias[co] = bias_vec[co];
    
    idx = 0;
    for (int co = 0; co < COUT; co++)
        for (int ro = 0; ro < ROUT; ro++)
            for (int ci = 0; ci < CIN; ci++)
                for (int ri = 0; ri < RIN; ri++)
                    for (int k = 0; k < 9; k++)
                        for (int d = 0; d < 4; d++)
                            kernel_expansion_idx[co][ro][ci][ri][k][d] = kernel_exp_idx_vec[idx++];
    
    idx = 0;
    for (int ri = 0; ri < RIN; ri++)
        for (int c = 0; c < CHARTS; c++)
            for (int h = 0; h < H_PADDED; h++)
                for (int w = 0; w < W_PADDED; w++)
                    reorder_idx[ri][c][h][w] = reorder_idx_vec[idx++];
    
    std::cout << "  Arrays prepared." << std::endl;
    
    // ==================== 3. 提取第0帧并保存输入 ====================
    std::cout << "\n[3] Extracting frame 0 and saving intermediate outputs..." << std::endl;
    
    // Frame 0 input: [Cin=1, Rin=1, charts=5, H=4, W=8]
    // 注意：input 的格式是 [TIME_STEPS][CIN][RIN][CHARTS][H][W]
    // 我们需要提取为 [CIN][RIN][CHARTS][H][W]
    static data_t frame0_input[CIN][RIN][CHARTS][H][W];
    for (int ci = 0; ci < CIN; ci++)
        for (int ri = 0; ri < RIN; ri++)
            for (int c = 0; c < CHARTS; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        frame0_input[ci][ri][c][h][w] = input[0][ci][ri][c][h][w];
    
    // 保存 frame0_input （需要 reshape 为 [RIN][CHARTS][H][W] 格式）
    static data_t frame0_input_save[RIN][CHARTS][H][W];
    for (int ri = 0; ri < RIN; ri++)
        for (int c = 0; c < CHARTS; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    frame0_input_save[ri][c][h][w] = frame0_input[0][ri][c][h][w];
    
    save_ico_tensor_4d<RIN, CHARTS, H, W>(
        DEBUG_DIR + "cpp_frame0_input.txt",
        frame0_input_save,
        "Frame 0 Input [1, 5, 4, 8]"
    );
    
    // ==================== 4. PadIco with SmoothVertices ====================
    // 步骤 1: 先对输入应用 SmoothVertices (process_vertices)
    // SmoothVertices 的逻辑：
    // 1. 先 CleanVertices（将顶点清零）
    // 2. 然后用邻居均值替换顶点
    
    static data_t input_after_smooth[CIN][RIN][CHARTS][H][W];
    
    // 复制输入
    for (int ci = 0; ci < CIN; ci++)
        for (int ri = 0; ri < RIN; ri++)
            for (int c = 0; c < CHARTS; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        input_after_smooth[ci][ri][c][h][w] = frame0_input[ci][ri][c][h][w];
    
    // CleanVertices: 将顶点位置清零
    // 对于 r=2: H=4, W=8
    // 顶点位置: (0,0) 和 (0, 2^r) = (0, 4)
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                input_after_smooth[ci][ri][c][0][0] = 0.0f;  // 北极
                input_after_smooth[ci][ri][c][0][H] = 0.0f;  // 南极 (W/2 = 4)
            }
        }
    }
    
    // SmoothVertices: 用邻居均值替换顶点
    // v1 (0,0) 的 5 个邻居: (chart, 1, 0), (chart, 1, 1), (chart, 0, 1), 
    //                      (chart-1, H-1, H), (chart-1, H-1, H-1)
    // v2 (0,H) 的 5 个邻居: (chart, 1, H), (chart, 1, H+1), (chart, 0, H+1),
    //                      (chart-1, H-1, W-1), (chart, 0, H-1)
    
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                int prev_c = (c - 1 + CHARTS) % CHARTS;
                
                // v1 (0,0) 的均值
                float sum_v1 = 0.0f;
                sum_v1 += frame0_input[ci][ri][c][1][0];
                sum_v1 += frame0_input[ci][ri][c][1][1];
                sum_v1 += frame0_input[ci][ri][c][0][1];
                sum_v1 += frame0_input[ci][ri][prev_c][H-1][H];    // chart-1, -1 -> H-1, 2^r -> H
                sum_v1 += frame0_input[ci][ri][prev_c][H-1][H-1];  // chart-1, -1 -> H-1, 2^r-1 -> H-1
                input_after_smooth[ci][ri][c][0][0] = sum_v1 / 5.0f;
                
                // v2 (0,H) 的均值
                float sum_v2 = 0.0f;
                sum_v2 += frame0_input[ci][ri][c][1][H];      // (chart, 1, 2^r)
                sum_v2 += frame0_input[ci][ri][c][1][(H+1)%W];  // (chart, 1, 2^r+1) 注意循环
                sum_v2 += frame0_input[ci][ri][c][0][(H+1)%W];  // (chart, 0, 2^r+1)
                sum_v2 += frame0_input[ci][ri][prev_c][H-1][W-1];  // (chart-1, -1, -1) -> (chart-1, H-1, W-1)
                sum_v2 += frame0_input[ci][ri][c][0][H-1];    // (chart, 0, 2^r-1)
                input_after_smooth[ci][ri][c][0][H] = sum_v2 / 5.0f;
            }
        }
    }
    
    // 步骤 2: 计算极点平滑值（用于 padding 后设置）
    // smooth_north_pole: 北极在 (H-1, 0) 位置
    // smooth_south_pole: 南极在 (0, W-1) 位置
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
    
    // 步骤 3: 执行 PadIco (使用 reorder_idx)
    static data_t padded[RIN][CHARTS][H_PADDED][W_PADDED];
    
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                for (int h = 0; h < H_PADDED; h++) {
                    for (int w = 0; w < W_PADDED; w++) {
                        int reorder_val = reorder_idx[ri][c][h][w];
                        
                        int src_chart = reorder_val / (H * W);
                        int remainder = reorder_val % (H * W);
                        int src_h = remainder / W;
                        int src_w = remainder % W;
                        
                        padded[ri][c][h][w] = input_after_smooth[ci][ri][src_chart][src_h][src_w];
                    }
                }
            }
        }
    }
    
    // 步骤 4: 设置极点平滑值
    // y[..., -1, 1] = smooth_north_pole  -> padded[ri][c][H_PADDED-1][1]
    // y[..., 1, -1] = smooth_south_pole  -> padded[ri][c][1][W_PADDED-1]
    for (int ri = 0; ri < RIN; ri++) {
        for (int c = 0; c < CHARTS; c++) {
            padded[ri][c][H_PADDED-1][1] = smooth_north_pole;
            padded[ri][c][1][W_PADDED-1] = smooth_south_pole;
        }
    }
    
    save_ico_tensor_4d<RIN, CHARTS, H_PADDED, W_PADDED>(
        DEBUG_DIR + "cpp_frame0_padded.txt",
        padded,
        "After PadIco [1, 5, 6, 10]"
    );
    
    // ==================== 5. Reshape 为 2D 卷积格式 ====================
    static data_t reshaped_input[CIN*RIN][CHARTS*H_PADDED][W_PADDED];
    
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            int c_idx = ci * RIN + ri;
            for (int chart = 0; chart < CHARTS; chart++) {
                for (int h = 0; h < H_PADDED; h++) {
                    int h_idx = chart * H_PADDED + h;
                    for (int w = 0; w < W_PADDED; w++) {
                        reshaped_input[c_idx][h_idx][w] = padded[ri][chart][h][w];
                    }
                }
            }
        }
    }
    
    save_tensor_3d<CIN*RIN, CHARTS*H_PADDED, W_PADDED>(
        DEBUG_DIR + "cpp_frame0_reshaped_input.txt",
        reshaped_input,
        "Reshaped Input [1, 30, 10]"
    );
    
    // ==================== 6. 构造卷积核 ====================
    // 省略卷积核输出（太大），直接执行卷积
    
    // ==================== 7. 执行完整的 Layer0 计算 ====================
    std::cout << "\n[4] Running full Layer0 forward pass..." << std::endl;
    
    static data_t output_full[TIME_STEPS][COUT][ROUT][CHARTS][H][W];
    
    conv_ico_layer0(input, weight, bias, kernel_expansion_idx, reorder_idx, output_full);
    
    // 提取第0帧输出
    static data_t frame0_output[COUT][ROUT][CHARTS][H][W];
    for (int co = 0; co < COUT; co++)
        for (int ro = 0; ro < ROUT; ro++)
            for (int c = 0; c < CHARTS; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        frame0_output[co][ro][c][h][w] = output_full[0][co][ro][c][h][w];
    
    save_output_5d<COUT, ROUT, CHARTS, H, W>(
        DEBUG_DIR + "cpp_frame0_final_output.txt",
        frame0_output,
        "Final Output [32, 6, 5, 4, 8]"
    );
    
    // ==================== 8. 总结 ====================
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "所有中间层数据已保存到: " << DEBUG_DIR << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}
