#include "ico_conv_layer1.hpp"
#include "../common/utils.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static std::string resolve_data_dir() {
    const std::vector<std::string> candidates = {
        "../hls_testdata/layer1/",
        "../../hls_testdata/layer1/",
        "../../../hls_testdata/layer1/",
        "../../../../hls_testdata/layer1/",
        "../../../../../hls_testdata/layer1/",
        "../../../../../../hls_testdata/layer1/",
        "../../../../../../../hls_testdata/layer1/"
    };
    for (size_t i = 0; i < candidates.size(); i++) {
        if (file_exists(candidates[i] + "input_rearranged.txt")) {
            return candidates[i];
        }
    }
    return "";
}

int main() {
    std::cout << "=== IcoConv Layer 1 C Verification ===" << std::endl;

    const std::string data_dir = resolve_data_dir();
    if (data_dir.empty()) {
        std::cerr << "Error: cannot locate hls_testdata/layer1" << std::endl;
        return -1;
    }
    std::cout << "Data dir: " << data_dir << std::endl;

    auto input_vec = read_txt_data(data_dir + "input_rearranged.txt");
    auto weight_vec = read_txt_data(data_dir + "weight.txt");
    auto bias_vec = read_txt_data(data_dir + "bias.txt");
    auto kernel_idx_vec = read_txt_data_int(data_dir + "kernel_expansion_idx.txt");
    auto reorder_idx_vec = read_txt_data_int(data_dir + "reorder_idx.txt");
    auto ref_output_vec = read_txt_data(data_dir + "output_layer1.txt");

    const size_t input_expected = (size_t)TIME_STEPS * CIN * RIN * CHARTS * H * W;
    const size_t weight_expected = (size_t)COUT * CIN * RIN * 7;
    const size_t bias_expected = (size_t)COUT;
    const size_t kernel_idx_expected = (size_t)COUT * ROUT * CIN * RIN * 9 * 4;
    const size_t reorder_idx_expected = (size_t)RIN * CHARTS * H_PADDED * W_PADDED;
    const size_t output_expected = (size_t)TIME_STEPS * COUT * ROUT * CHARTS * H * W;

    if (input_vec.size() != input_expected) {
        std::cerr << "Input size mismatch. expected=" << input_expected << " got=" << input_vec.size() << std::endl;
        return -2;
    }
    if (weight_vec.size() != weight_expected) {
        std::cerr << "Weight size mismatch. expected=" << weight_expected << " got=" << weight_vec.size() << std::endl;
        return -3;
    }
    if (bias_vec.size() != bias_expected) {
        std::cerr << "Bias size mismatch. expected=" << bias_expected << " got=" << bias_vec.size() << std::endl;
        return -4;
    }
    if (kernel_idx_vec.size() != kernel_idx_expected) {
        std::cerr << "Kernel index size mismatch. expected=" << kernel_idx_expected << " got=" << kernel_idx_vec.size() << std::endl;
        return -5;
    }
    if (reorder_idx_vec.size() != reorder_idx_expected) {
        std::cerr << "Reorder index size mismatch. expected=" << reorder_idx_expected << " got=" << reorder_idx_vec.size() << std::endl;
        return -6;
    }

    static data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W];
    static data_t weight[COUT][CIN][RIN][7];
    static data_t bias[COUT];
    static int kernel_idx[COUT][ROUT][CIN][RIN][9][4];
    static int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED];
    static data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W];

    size_t idx = 0;
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

    for (int co = 0; co < COUT; co++) bias[co] = bias_vec[co];

    idx = 0;
    for (int co = 0; co < COUT; co++)
        for (int ro = 0; ro < ROUT; ro++)
            for (int ci = 0; ci < CIN; ci++)
                for (int ri = 0; ri < RIN; ri++)
                    for (int k = 0; k < 9; k++)
                        for (int d = 0; d < 4; d++)
                            kernel_idx[co][ro][ci][ri][k][d] = kernel_idx_vec[idx++];

    idx = 0;
    for (int ri = 0; ri < RIN; ri++)
        for (int c = 0; c < CHARTS; c++)
            for (int h = 0; h < H_PADDED; h++)
                for (int w = 0; w < W_PADDED; w++)
                    reorder_idx[ri][c][h][w] = reorder_idx_vec[idx++];

    std::cout << "Running conv_ico_layer1..." << std::endl;
    conv_ico_layer1(input, weight, bias, kernel_idx, reorder_idx, output);

    std::vector<float> out_flat;
    out_flat.reserve(output_expected);
    for (int t = 0; t < TIME_STEPS; t++)
        for (int co = 0; co < COUT; co++)
            for (int ro = 0; ro < ROUT; ro++)
                for (int c = 0; c < CHARTS; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            out_flat.push_back(output[t][co][ro][c][h][w]);

    print_stats("Layer1 Output", out_flat);

    if (!ref_output_vec.empty()) {
        if (ref_output_vec.size() != output_expected) {
            std::cerr << "Reference output size mismatch. expected=" << output_expected
                      << " got=" << ref_output_vec.size() << std::endl;
            return -7;
        }
        float max_err = max_error(out_flat, ref_output_vec);
        float rms_err = rmse(out_flat, ref_output_vec);
        std::cout << "Max Error: " << max_err << std::endl;
        std::cout << "RMSE: " << rms_err << std::endl;
        if (max_err < 1e-3f) {
            std::cout << "PASS" << std::endl;
        } else {
            std::cout << "FAIL" << std::endl;
        }
    } else {
        std::cout << "Reference output file not found or empty, skip compare." << std::endl;
    }

    return 0;
}
