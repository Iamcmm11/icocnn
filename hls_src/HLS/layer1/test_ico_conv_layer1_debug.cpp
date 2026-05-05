#include "ico_conv_layer1.hpp"
#include "../common/utils.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

template <int D0, int D1, int D2, int D3, int D4>
static void save_tensor_5d(
    const std::string& file,
    data_t (&arr)[D0][D1][D2][D3][D4],
    const std::string& name
) {
    std::ofstream f(file.c_str());
    if (!f.is_open()) {
        std::cerr << "Error: cannot write " << file << std::endl;
        return;
    }

    double min_v = std::numeric_limits<double>::max();
    double max_v = std::numeric_limits<double>::lowest();
    double sum = 0.0;
    size_t cnt = 0;
    for (int i0 = 0; i0 < D0; i0++) {
        for (int i1 = 0; i1 < D1; i1++) {
            for (int i2 = 0; i2 < D2; i2++) {
                for (int i3 = 0; i3 < D3; i3++) {
                    for (int i4 = 0; i4 < D4; i4++) {
                        double v = arr[i0][i1][i2][i3][i4];
                        if (v < min_v) min_v = v;
                        if (v > max_v) max_v = v;
                        sum += v;
                        cnt++;
                    }
                }
            }
        }
    }

    f << std::fixed << std::setprecision(8);
    f << "# " << name << "\n";
    f << "# Shape: (" << D0 << ", " << D1 << ", " << D2 << ", " << D3 << ", " << D4 << ")\n";
    f << "# Min: " << min_v << ", Max: " << max_v << ", Mean: " << (sum / (double)cnt) << "\n";
    f << "#" << std::string(70, '=') << "\n\n";

    for (int i0 = 0; i0 < D0; i0++) {
        for (int i1 = 0; i1 < D1; i1++) {
            for (int i2 = 0; i2 < D2; i2++) {
                f << "# [" << i0 << ", " << i1 << ", chart" << i2 << "] - Shape: (" << D3 << ", " << D4 << ")\n";
                for (int i3 = 0; i3 < D3; i3++) {
                    f << "  ";
                    for (int i4 = 0; i4 < D4; i4++) {
                        f << arr[i0][i1][i2][i3][i4];
                        if (i4 + 1 < D4) f << "  ";
                    }
                    f << "\n";
                }
                f << "\n";
            }
        }
    }
}

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

    save_tensor_5d(
        out_dir + "cpp_frame0_input.txt",
        frame0,
        "Frame0 Input [CIN,RIN,CHARTS,H,W]"
    );

    static data_t padded[CIN][RIN][CHARTS][H_PADDED][W_PADDED];
    pad_ico(frame0, reorder_idx, padded);

    save_tensor_5d(
        out_dir + "cpp_frame0_padded.txt",
        padded,
        "After PadIco [CIN,RIN,CHARTS,H_PADDED,W_PADDED]"
    );

    static data_t out[TIME_STEPS][COUT][ROUT][CHARTS][H][W];
    conv_ico_layer1(input, weight, bias, kernel_idx, reorder_idx, out);

    save_tensor_5d(
        out_dir + "cpp_frame0_final_output.txt",
        out[0],
        "Frame0 Final Output [COUT,ROUT,CHARTS,H,W]"
    );

    std::cout << "Saved layer1 debug intermediates to: " << out_dir << std::endl;
    return 0;
}
