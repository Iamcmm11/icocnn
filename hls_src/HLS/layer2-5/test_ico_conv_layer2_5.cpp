#include "ico_conv_layer2_5.hpp"
#include "../common/utils.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef ICO_LAYER2_5_FORCE_FILE_DATA
#define ICO_LAYER2_5_FORCE_FILE_DATA 0
#endif

#ifndef ICO_LAYER2_5_FORCE_SYNTHETIC_DATA
#define ICO_LAYER2_5_FORCE_SYNTHETIC_DATA 0
#endif

#if (TIME_STEPS == 52 && CIN == 32 && COUT == 32) || ICO_LAYER2_5_FORCE_FILE_DATA
#define ICO_LAYER2_5_DEFAULT_FILE_DATA 1
#else
#define ICO_LAYER2_5_DEFAULT_FILE_DATA 0
#endif

static bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static int parse_layer_id(int argc, char** argv) {
    if (argc < 2) return 2;
    const int layer_id = std::atoi(argv[1]);
    if (layer_id < 2 || layer_id > 5) return 2;
    return layer_id;
}

static std::string normalize_data_dir(const std::string& path) {
    if (path.empty()) return path;
    const char last = path[path.size() - 1];
    if (last == '/' || last == '\\') return path;
    return path + "/";
}

static std::string resolve_data_dir(int layer_id, int argc, char** argv) {
    if (argc >= 3) {
        return normalize_data_dir(argv[2]);
    }

    const char* env_dir = std::getenv("ICO_LAYER2_5_DATA_DIR");
    if (env_dir != NULL && env_dir[0] != '\0') {
        return normalize_data_dir(env_dir);
    }

    const std::string suffix = "layer" + std::to_string(layer_id) + "/";
    const std::vector<std::string> candidates = {
        "../hls_testdata/layer2-5/" + suffix,
        "../../hls_testdata/layer2-5/" + suffix,
        "../../../hls_testdata/layer2-5/" + suffix,
        "../../../../hls_testdata/layer2-5/" + suffix,
        "../../../../../hls_testdata/layer2-5/" + suffix,
        "../../../../../../hls_testdata/layer2-5/" + suffix,
        "../../../../../../../hls_testdata/layer2-5/" + suffix
    };

    for (size_t i = 0; i < candidates.size(); i++) {
        if (file_exists(candidates[i] + "input_rearranged.txt")) {
            return candidates[i];
        }
    }
    return "";
}

static bool should_use_file_data(int argc) {
#if ICO_LAYER2_5_FORCE_SYNTHETIC_DATA
    return false;
#else
    const char* env_dir = std::getenv("ICO_LAYER2_5_DATA_DIR");
    if (argc >= 3 || (env_dir != NULL && env_dir[0] != '\0')) {
        return true;
    }
    return ICO_LAYER2_5_DEFAULT_FILE_DATA != 0;
#endif
}

static void fill_synthetic_input(data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W]) {
    for (int t = 0; t < TIME_STEPS; t++)
        for (int ci = 0; ci < CIN; ci++)
            for (int ri = 0; ri < RIN; ri++)
                for (int c = 0; c < CHARTS; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            input[t][ci][ri][c][h][w] =
                                0.01f * static_cast<float>((t + 1) + (ci % 5) + (ri % 3)) +
                                0.001f * static_cast<float>(c * H * W + h * W + w);
}

static void fill_synthetic_weight(data_t weight[COUT][CIN][RIN][7], data_t bias[COUT]) {
    for (int co = 0; co < COUT; co++) {
        bias[co] = 0.001f * static_cast<float>(co - (COUT / 2));
        for (int ci = 0; ci < CIN; ci++)
            for (int ri = 0; ri < RIN; ri++)
                for (int k = 0; k < 7; k++)
                    weight[co][ci][ri][k] =
                        0.0005f * static_cast<float>(((co + 1) * (ci + 2) + ri + k) % 17 - 8);
    }
}

static void fill_synthetic_kernel_idx(int kernel_idx[COUT][ROUT][CIN][RIN][9][4]) {
    for (int co = 0; co < COUT; co++)
        for (int ro = 0; ro < ROUT; ro++)
            for (int ci = 0; ci < CIN; ci++)
                for (int ri = 0; ri < RIN; ri++)
                    for (int k = 0; k < 9; k++) {
                        kernel_idx[co][ro][ci][ri][k][0] = co;
                        kernel_idx[co][ro][ci][ri][k][1] = ci;
                        kernel_idx[co][ro][ci][ri][k][2] = ri;
                        kernel_idx[co][ro][ci][ri][k][3] = k % 7;
                    }
}

static void fill_synthetic_reorder_idx(int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED]) {
    for (int ri = 0; ri < RIN; ri++)
        for (int c = 0; c < CHARTS; c++)
            for (int hp = 0; hp < H_PADDED; hp++)
                for (int wp = 0; wp < W_PADDED; wp++) {
                    int src_h = hp - 1;
                    int src_w = wp - 1;
                    if (src_h < 0) src_h = 0;
                    if (src_h >= H) src_h = H - 1;
                    if (src_w < 0) src_w = 0;
                    if (src_w >= W) src_w = W - 1;
                    reorder_idx[ri][c][hp][wp] = ((ri * CHARTS + c) * H + src_h) * W + src_w;
                }
}

int main(int argc, char** argv) {
    const int layer_id = parse_layer_id(argc, argv);
    std::cout << "=== IcoConv Layer2-5 C Verification ===" << std::endl;
    std::cout << "Selected layer: " << layer_id << std::endl;
    std::cout << "Configured shape: T=" << TIME_STEPS
              << " CIN=" << CIN << " COUT=" << COUT
              << " RIN=" << RIN << " ROUT=" << ROUT << std::endl;

    const size_t input_expected = (size_t)TIME_STEPS * CIN * RIN * CHARTS * H * W;
    const size_t weight_expected = (size_t)COUT * CIN * RIN * 7;
    const size_t bias_expected = (size_t)COUT;
    const size_t kernel_idx_expected = (size_t)COUT * ROUT * CIN * RIN * 9 * 4;
    const size_t reorder_idx_expected = (size_t)RIN * CHARTS * H_PADDED * W_PADDED;
    const size_t output_expected = (size_t)TIME_STEPS * COUT * ROUT * CHARTS * H * W;

    static data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W];
    static data_t weight[COUT][CIN][RIN][7];
    static data_t bias[COUT];
    static int kernel_idx[COUT][ROUT][CIN][RIN][9][4];
    static int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED];
    static data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W];

    std::vector<float> ref_output_vec;
    if (should_use_file_data(argc)) {
        const std::string data_dir = resolve_data_dir(layer_id, argc, argv);
        if (data_dir.empty()) {
            std::cerr << "Error: cannot locate hls_testdata/layer2-5/layer" << layer_id << std::endl;
            return -1;
        }
        if (!file_exists(data_dir + "input_rearranged.txt")) {
            std::cerr << "Error: data dir does not contain input_rearranged.txt: " << data_dir << std::endl;
            return -1;
        }
        std::cout << "Data dir: " << data_dir << std::endl;

        auto input_vec = read_txt_data(data_dir + "input_rearranged.txt");
        auto weight_vec = read_txt_data(data_dir + "weight.txt");
        auto bias_vec = read_txt_data(data_dir + "bias.txt");
        auto kernel_idx_vec = read_txt_data_int(data_dir + "kernel_expansion_idx.txt");
        auto reorder_idx_vec = read_txt_data_int(data_dir + "reorder_idx.txt");
        ref_output_vec = read_txt_data(data_dir + "output.txt");

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
    } else {
        std::cout << "Data mode: deterministic synthetic smoke" << std::endl;
        fill_synthetic_input(input);
        fill_synthetic_weight(weight, bias);
        fill_synthetic_kernel_idx(kernel_idx);
        fill_synthetic_reorder_idx(reorder_idx);
    }

    std::cout << "Running conv_ico_layer2_5..." << std::endl;
    conv_ico_layer2_5(input, weight, bias, kernel_idx, reorder_idx, output);

    std::vector<float> out_flat;
    out_flat.reserve(output_expected);
    for (int t = 0; t < TIME_STEPS; t++)
        for (int co = 0; co < COUT; co++)
            for (int ro = 0; ro < ROUT; ro++)
                for (int c = 0; c < CHARTS; c++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            out_flat.push_back(output[t][co][ro][c][h][w]);

    print_stats("Layer2-5 Output", out_flat);

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
        std::cout << (max_err < 1e-3f ? "PASS" : "FAIL") << std::endl;
    } else {
        std::cout << "Reference output file not found or empty, skip compare." << std::endl;
    }

    return 0;
}
