#include "ico_conv_layer0.hpp"
#include "utils.hpp"
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::cout << "=== IcoConv Layer 0 HLS Testbench ===" << std::endl;
    
    const std::string data_dir = "../hls_testdata/layer0/";
    
    // ==================== 1. 读取输入数据 ====================
    std::cout << "\n[1] Loading input data..." << std::endl;
    
    auto input_rearranged_vec = read_txt_data(data_dir + "input_rearranged.txt");
    if (input_rearranged_vec.empty()) {
        std::cerr << "Error: Failed to load input_rearranged.txt" << std::endl;
        return -1;
    }
    print_stats("Input", input_rearranged_vec);
    
    // ==================== 2. 读取权重和偏置 ====================
    std::cout << "\n[2] Loading weights and bias..." << std::endl;
    
    auto weight_vec = read_txt_data(data_dir + "weight.txt");
    auto bias_vec = read_txt_data(data_dir + "bias.txt");
    
    if (weight_vec.size() != COUT * CIN * RIN * 7) {
        std::cerr << "Error: Weight size mismatch! Expected " << (COUT * CIN * RIN * 7) 
                  << " got " << weight_vec.size() << std::endl;
        return -1;
    }
    if (bias_vec.size() != COUT) {
        std::cerr << "Error: Bias size mismatch! Expected " << COUT 
                  << " got " << bias_vec.size() << std::endl;
        return -1;
    }
    
    print_stats("Weight", weight_vec);
    print_stats("Bias", bias_vec);
    
    // ==================== 3. 读取索引表 ====================
    std::cout << "\n[3] Loading index tables..." << std::endl;
    
    auto kernel_exp_idx_vec = read_txt_data_int(data_dir + "kernel_expansion_idx.txt");
    auto reorder_idx_vec = read_txt_data_int(data_dir + "reorder_idx.txt");
    
    std::cout << "Kernel expansion idx size: " << kernel_exp_idx_vec.size() << std::endl;
    std::cout << "Reorder idx size: " << reorder_idx_vec.size() << std::endl;
    
    // ==================== 4. 分配数组并填充数据 ====================
    std::cout << "\n[4] Preparing arrays..." << std::endl;
    
    static data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W];
    static data_t weight[COUT][CIN][RIN][7];
    static data_t bias[COUT];
    static int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4];
    static int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED];
    static data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W];
    
    // 填充 input
    int idx = 0;
    for (int t = 0; t < TIME_STEPS; t++) {
        for (int ci = 0; ci < CIN; ci++) {
            for (int ri = 0; ri < RIN; ri++) {
                for (int c = 0; c < CHARTS; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            if (idx < input_rearranged_vec.size()) {
                                input[t][ci][ri][c][h][w] = input_rearranged_vec[idx++];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 填充 weight
    idx = 0;
    for (int co = 0; co < COUT; co++) {
        for (int ci = 0; ci < CIN; ci++) {
            for (int ri = 0; ri < RIN; ri++) {
                for (int k = 0; k < 7; k++) {
                    if (idx < weight_vec.size()) {
                        weight[co][ci][ri][k] = weight_vec[idx++];
                    }
                }
            }
        }
    }
    
    // 填充 bias
    for (int co = 0; co < COUT; co++) {
        if (co < bias_vec.size()) {
            bias[co] = bias_vec[co];
        }
    }
    
    // 填充 kernel_expansion_idx
    idx = 0;
    for (int co = 0; co < COUT; co++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int ci = 0; ci < CIN; ci++) {
                for (int ri = 0; ri < RIN; ri++) {
                    for (int k = 0; k < 9; k++) {
                        for (int d = 0; d < 4; d++) {
                            if (idx < kernel_exp_idx_vec.size()) {
                                kernel_expansion_idx[co][ro][ci][ri][k][d] = kernel_exp_idx_vec[idx++];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 填充 reorder_idx
    idx = 0;
    for (int ri = 0; ri < RIN; ri++) {
        for (int c = 0; c < CHARTS; c++) {
            for (int h = 0; h < H_PADDED; h++) {
                for (int w = 0; w < W_PADDED; w++) {
                    if (idx < reorder_idx_vec.size()) {
                        reorder_idx[ri][c][h][w] = reorder_idx_vec[idx++];
                    }
                }
            }
        }
    }
    
    std::cout << "Arrays prepared successfully." << std::endl;
    
    // ==================== 5. 执行 HLS 函数 ====================
    std::cout << "\n[5] Running IcoConv Layer 0..." << std::endl;
    
    conv_ico_layer0(input, weight, bias, kernel_expansion_idx, reorder_idx, output);
    
    std::cout << "IcoConv Layer 0 finished." << std::endl;
    
    // ==================== 6. 读取参考输出并对比 ====================
    std::cout << "\n[6] Comparing with reference output..." << std::endl;
    
    auto ref_output_vec = read_txt_data(data_dir + "output_layer0.txt");
    if (ref_output_vec.empty()) {
        std::cerr << "Warning: No reference output found." << std::endl;
    } else {
        // Flatten HLS output
        std::vector<float> hls_output_vec;
        for (int t = 0; t < TIME_STEPS; t++) {
            for (int co = 0; co < COUT; co++) {
                for (int ro = 0; ro < ROUT; ro++) {
                    for (int c = 0; c < CHARTS; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                hls_output_vec.push_back(output[t][co][ro][c][h][w]);
                            }
                        }
                    }
                }
            }
        }
        
        print_stats("HLS Output", hls_output_vec);
        print_stats("Reference Output", ref_output_vec);
        
        float max_err = max_error(hls_output_vec, ref_output_vec);
        float rms_err = rmse(hls_output_vec, ref_output_vec);
        
        std::cout << "\n=== Verification Results ===" << std::endl;
        std::cout << "Max Error: " << max_err << std::endl;
        std::cout << "RMSE: " << rms_err << std::endl;
        
        if (max_err < 1e-3) {
            std::cout << "\n✓ PASS: HLS output matches PyTorch reference!" << std::endl;
        } else {
            std::cout << "\n✗ FAIL: Significant difference detected!" << std::endl;
        }
    }
    
    return 0;
}
