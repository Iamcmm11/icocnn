#include "ico_conv_layer1.hpp"
#include "../utils.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void save_flat(const std::string& file, const std::vector<float>& v, const std::string& shape) {
    std::ofstream f(file.c_str());
    f << "# Shape: " << shape << "\n";
    for (size_t i = 0; i < v.size(); i++) {
        f << v[i] << "\n";
    }
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static std::string resolve_data_dir() {
    const std::vector<std::string> candidates = {
        "../../hls_testdata/layer1/",
        "../../../hls_testdata/layer1/",
        "../hls_testdata/layer1/"
    };
    for (size_t i = 0; i < candidates.size(); i++) {
        if (file_exists(candidates[i] + "input_rearranged.txt")) return candidates[i];
    }
    return "";
}

int main() {
    const std::string data_dir = resolve_data_dir();
    if (data_dir.empty()) {
        std::cerr << "Error: cannot locate hls_testdata/layer1" << std::endl;
        return -1;
    }

    const std::string out_dir = data_dir + "debug_intermediate_cpp/";

    auto input_vec = read_txt_data(data_dir + "input_rearranged.txt");
    auto weight_vec = read_txt_data(data_dir + "weight.txt");
    auto bias_vec = read_txt_data(data_dir + "bias.txt");
    auto kernel_idx_vec = read_txt_data_int(data_dir + "kernel_expansion_idx.txt");
    auto reorder_idx_vec = read_txt_data_int(data_dir + "reorder_idx.txt");

    static data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W];
    static data_t weight[COUT][CIN][RIN][7];
    static data_t bias[COUT];
    static int kernel_idx[COUT][ROUT][CIN][RIN][9][4];
    static int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED];

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

    static data_t frame0[CIN][RIN][CHARTS][H][W];
    for (int ci = 0; ci < CIN; ci++)
        for (int ri = 0; ri < RIN; ri++)
            for (int c = 0; c < CHARTS; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        frame0[ci][ri][c][h][w] = input[0][ci][ri][c][h][w];

    std::vector<float> frame0_flat;
    frame0_flat.reserve((size_t)CIN * RIN * CHARTS * H * W);
    for (int ci = 0; ci < CIN; ci++)
        for (int ri = 0; ri < RIN; ri++)
            for (int c = 0; c < CHARTS; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        frame0_flat.push_back(frame0[ci][ri][c][h][w]);
    save_flat(out_dir + "cpp_frame0_input.txt", frame0_flat, "[CIN,RIN,CHARTS,H,W]");

    static data_t padded[CIN][RIN][CHARTS][H_PADDED][W_PADDED];
    pad_ico(frame0, reorder_idx, padded);

    std::vector<float> padded_flat;
    padded_flat.reserve((size_t)CIN * RIN * CHARTS * H_PADDED * W_PADDED);
    for (int ci = 0; ci < CIN; ci++)
        for (int ri = 0; ri < RIN; ri++)
            for (int c = 0; c < CHARTS; c++)
                for (int h = 0; h < H_PADDED; h++)
                    for (int w = 0; w < W_PADDED; w++)
                        padded_flat.push_back(padded[ci][ri][c][h][w]);
    save_flat(out_dir + "cpp_frame0_padded.txt", padded_flat, "[CIN,RIN,CHARTS,H_PADDED,W_PADDED]");

    static data_t out[TIME_STEPS][COUT][ROUT][CHARTS][H][W];
    conv_ico_layer1(input, weight, bias, kernel_idx, reorder_idx, out);

    std::vector<float> out0_flat;
    out0_flat.reserve((size_t)COUT * ROUT * CHARTS * H * W);
    for (int co = 0; co < COUT; co++)
        for (int ro = 0; ro < ROUT; ro++)
            for (int c = 0; c < CHARTS; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        out0_flat.push_back(out[0][co][ro][c][h][w]);
    save_flat(out_dir + "cpp_frame0_final_output.txt", out0_flat, "[COUT,ROUT,CHARTS,H,W]");

    std::cout << "Saved layer1 debug intermediates to: " << out_dir << std::endl;
    return 0;
}
