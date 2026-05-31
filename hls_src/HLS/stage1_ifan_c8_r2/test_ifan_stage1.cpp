#include "ifan_stage1.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

static data_t g_input[IFAN_IN_CHANNELS][IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
static IfanStage1Weights g_weights;
static int g_reorder_r2_stem[1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2];
static int g_reorder_r2_main[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2];
static int g_reorder_r1[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1 + 2][IFAN_W_R1 + 2];
static int g_kernel_idx_stem[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][1][1][9][4];
static int g_kernel_idx_main[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4];
static data_t g_output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_golden[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];

template <typename T>
static bool read_flat_file(const std::string &path, T *dst, std::size_t count) {
    std::ifstream f(path.c_str());
    if (!f) {
        return false;
    }

    std::size_t index = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        while (iss && index < count) {
            T value;
            if (!(iss >> value)) {
                break;
            }
            dst[index++] = value;
        }
    }

    if (index != count) {
        std::cerr << "FAIL: " << path << " contains " << index
                  << " values, expected " << count << "\n";
        return false;
    }
    return true;
}

template <typename T>
static bool read_required(const std::string &data_dir, const std::string &name, T *dst, std::size_t count) {
    const std::string path = data_dir + "/" + name;
    if (!read_flat_file(path, dst, count)) {
        std::cerr << "Missing or invalid Stage-1 data file: " << path << "\n";
        return false;
    }
    return true;
}

static int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static int encode_reorder(int ri, int chart, int h, int w, int height, int width) {
    return (((ri * IFAN_CHARTS + chart) * height + h) * width + w);
}

static void init_reorder_tables() {
    for (int ch = 0; ch < IFAN_CHARTS; ch++) {
        for (int h = 0; h < IFAN_H_R2 + 2; h++) {
            for (int w = 0; w < IFAN_W_R2 + 2; w++) {
                const int src_h = clamp_int(h - 1, 0, IFAN_H_R2 - 1);
                const int src_w = clamp_int(w - 1, 0, IFAN_W_R2 - 1);
                g_reorder_r2_stem[0][ch][h][w] = encode_reorder(0, ch, src_h, src_w, IFAN_H_R2, IFAN_W_R2);
                for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                    g_reorder_r2_main[ri][ch][h][w] = encode_reorder(ri, ch, src_h, src_w, IFAN_H_R2, IFAN_W_R2);
                }
            }
        }
    }

    for (int ri = 0; ri < IFAN_R_FULL; ri++) {
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1 + 2; h++) {
                for (int w = 0; w < IFAN_W_R1 + 2; w++) {
                    const int src_h = clamp_int(h - 1, 0, IFAN_H_R1 - 1);
                    const int src_w = clamp_int(w - 1, 0, IFAN_W_R1 - 1);
                    g_reorder_r1[ri][ch][h][w] = encode_reorder(ri, ch, src_h, src_w, IFAN_H_R1, IFAN_W_R1);
                }
            }
        }
    }
}

static void init_kernel_tables() {
    for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
        for (int ro = 0; ro < IFAN_R_FULL; ro++) {
            for (int k = 0; k < 9; k++) {
                g_kernel_idx_stem[co][ro][0][0][k][0] = co;
                g_kernel_idx_stem[co][ro][0][0][k][1] = 0;
                g_kernel_idx_stem[co][ro][0][0][k][2] = 0;
                g_kernel_idx_stem[co][ro][0][0][k][3] = k % IFAN_KERNEL_NEIGHBORS;
            }
            for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
                for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                    for (int k = 0; k < 9; k++) {
                        g_kernel_idx_main[co][ro][ci][ri][k][0] = co;
                        g_kernel_idx_main[co][ro][ci][ri][k][1] = ci;
                        g_kernel_idx_main[co][ro][ci][ri][k][2] = ri;
                        g_kernel_idx_main[co][ro][ci][ri][k][3] = k % IFAN_KERNEL_NEIGHBORS;
                    }
                }
            }
        }
    }
}

static void init_weights() {
    std::memset(&g_weights, 0, sizeof(g_weights));

    for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
        g_weights.phat_stem_b[co] = 0.001f * (co + 1);
        g_weights.lms_stem_b[co] = -0.001f * (co + 1);
        for (int k = 0; k < IFAN_KERNEL_NEIGHBORS; k++) {
            g_weights.phat_stem_w[co][0][0][k] = 0.002f * ((co + k) % 3 + 1);
            g_weights.lms_stem_w[co][0][0][k] = 0.0015f * ((co + k) % 5 + 1);
        }
    }

    for (int layer = 0; layer < 2; layer++) {
        for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
            g_weights.phat_res_b[layer][co] = 0.0002f * (layer + 1);
            g_weights.lms_res_b[layer][co] = -0.0002f * (layer + 1);
            g_weights.attn_b[layer][co] = 0.0001f * (co + 1);
            for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
                for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                    for (int k = 0; k < IFAN_KERNEL_NEIGHBORS; k++) {
                        const data_t v = 0.00025f * ((co + ci + ri + k + layer) % 7 - 3);
                        g_weights.phat_res_w[layer][co][ci][ri][k] = v;
                        g_weights.lms_res_w[layer][co][ci][ri][k] = -v;
                        g_weights.attn_w[layer][co][ci][ri][k] = 0.00015f * ((co + ci + k) % 5 - 2);
                    }
                }
            }
        }
    }

    for (int b = 0; b < 4; b++) {
        for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
            g_weights.fusion_b[b][co] = 0.0001f * (b + 1);
            g_weights.fusion_temporal_b[b][co] = 0.00005f * (co + 1);
            for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
                for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                    for (int k = 0; k < IFAN_KERNEL_NEIGHBORS; k++) {
                        g_weights.fusion_w[b][co][ci][ri][k] = 0.0002f * ((b + co + ci + ri + k) % 9 - 4);
                    }
                }
                for (int k = 0; k < IFAN_TEMPORAL_KERNEL; k++) {
                    g_weights.fusion_temporal_w[b][co][ci][k] = 0.0003f * ((b + co + ci + k) % 7 - 3);
                }
            }
        }
    }

    for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
        g_weights.final_b[co] = 0.0001f * (co + 1);
        g_weights.final_temporal_b[co] = 0.0f;
        for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
            for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                for (int k = 0; k < IFAN_KERNEL_NEIGHBORS; k++) {
                    g_weights.final_w[co][ci][ri][k] = 0.00015f * ((co + ci + ri + k) % 7 - 3);
                }
            }
            for (int k = 0; k < IFAN_TEMPORAL_KERNEL; k++) {
                g_weights.final_temporal_w[co][ci][k] = 0.0002f * ((co + ci + k) % 5 - 2);
            }
        }
    }

    for (int n = 0; n < 16; n++) {
        for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
            g_weights.norm_gamma[n][c] = 1.0f;
            g_weights.norm_beta[n][c] = 0.0f;
        }
    }
}

static void init_input() {
    for (int c = 0; c < IFAN_IN_CHANNELS; c++) {
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                for (int h = 0; h < IFAN_H_R2; h++) {
                    for (int w = 0; w < IFAN_W_R2; w++) {
                        g_input[c][t][ch][h][w] =
                            0.01f * static_cast<data_t>((c + 1) * 3 + t + ch) +
                            0.001f * static_cast<data_t>(h * IFAN_W_R2 + w);
                    }
                }
            }
        }
    }
}

static bool init_real_data_from(const std::string &data_dir) {
    if (!read_flat_file(data_dir + "/stage1_input.txt",
                        &g_input[0][0][0][0][0],
                        IFAN_IN_CHANNELS * IFAN_STAGE1_T * IFAN_CHARTS * IFAN_H_R2 * IFAN_W_R2)) {
        return false;
    }

    bool ok = true;
    ok = ok && read_required(data_dir, "final_head_logits.txt",
                             &g_golden[0][0][0][0][0][0],
                             IFAN_STAGE1_T * IFAN_BRANCH_CHANNELS * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);

    ok = ok && read_required(data_dir, "weights/phat_stem_w.txt", &g_weights.phat_stem_w[0][0][0][0], 8 * 1 * 1 * 7);
    ok = ok && read_required(data_dir, "weights/phat_stem_b.txt", &g_weights.phat_stem_b[0], 8);
    ok = ok && read_required(data_dir, "weights/lms_stem_w.txt", &g_weights.lms_stem_w[0][0][0][0], 8 * 1 * 1 * 7);
    ok = ok && read_required(data_dir, "weights/lms_stem_b.txt", &g_weights.lms_stem_b[0], 8);

    ok = ok && read_required(data_dir, "weights/phat_res_w.txt", &g_weights.phat_res_w[0][0][0][0][0], 2 * 8 * 8 * 6 * 7);
    ok = ok && read_required(data_dir, "weights/phat_res_b.txt", &g_weights.phat_res_b[0][0], 2 * 8);
    ok = ok && read_required(data_dir, "weights/lms_res_w.txt", &g_weights.lms_res_w[0][0][0][0][0], 2 * 8 * 8 * 6 * 7);
    ok = ok && read_required(data_dir, "weights/lms_res_b.txt", &g_weights.lms_res_b[0][0], 2 * 8);

    ok = ok && read_required(data_dir, "weights/attn_w.txt", &g_weights.attn_w[0][0][0][0][0], 2 * 8 * 8 * 6 * 7);
    ok = ok && read_required(data_dir, "weights/attn_b.txt", &g_weights.attn_b[0][0], 2 * 8);

    ok = ok && read_required(data_dir, "weights/fusion_w.txt", &g_weights.fusion_w[0][0][0][0][0], 4 * 8 * 8 * 6 * 7);
    ok = ok && read_required(data_dir, "weights/fusion_b.txt", &g_weights.fusion_b[0][0], 4 * 8);
    ok = ok && read_required(data_dir, "weights/fusion_temporal_w.txt", &g_weights.fusion_temporal_w[0][0][0][0], 4 * 8 * 8 * 5);
    ok = ok && read_required(data_dir, "weights/fusion_temporal_b.txt", &g_weights.fusion_temporal_b[0][0], 4 * 8);

    ok = ok && read_required(data_dir, "weights/final_w.txt", &g_weights.final_w[0][0][0][0], 8 * 8 * 6 * 7);
    ok = ok && read_required(data_dir, "weights/final_b.txt", &g_weights.final_b[0], 8);
    ok = ok && read_required(data_dir, "weights/final_temporal_w.txt", &g_weights.final_temporal_w[0][0][0], 8 * 8 * 5);
    ok = ok && read_required(data_dir, "weights/final_temporal_b.txt", &g_weights.final_temporal_b[0], 8);

    ok = ok && read_required(data_dir, "weights/norm_gamma.txt", &g_weights.norm_gamma[0][0], 16 * 8);
    ok = ok && read_required(data_dir, "weights/norm_beta.txt", &g_weights.norm_beta[0][0], 16 * 8);

    ok = ok && read_required(data_dir, "geometry/reorder_r2_stem.txt", &g_reorder_r2_stem[0][0][0][0], 1 * 5 * 6 * 10);
    ok = ok && read_required(data_dir, "geometry/reorder_r2_main.txt", &g_reorder_r2_main[0][0][0][0], 6 * 5 * 6 * 10);
    ok = ok && read_required(data_dir, "geometry/reorder_r1.txt", &g_reorder_r1[0][0][0][0], 6 * 5 * 4 * 6);
    ok = ok && read_required(data_dir, "geometry/kernel_idx_stem.txt", &g_kernel_idx_stem[0][0][0][0][0][0], 8 * 6 * 1 * 1 * 9 * 4);
    ok = ok && read_required(data_dir, "geometry/kernel_idx_main.txt", &g_kernel_idx_main[0][0][0][0][0][0], 8 * 6 * 8 * 6 * 9 * 4);
    return ok;
}

static bool init_real_data() {
    const std::string candidates[] = {
        "../../../hls_testdata/stage1_ifan_c8_r2/scene_1_t6",
        "hls_testdata/stage1_ifan_c8_r2/scene_1_t6"
    };
    for (std::size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        if (init_real_data_from(candidates[i])) {
            std::cout << "Loaded real Stage-1 data: " << candidates[i] << "\n";
            return true;
        }
    }
    return false;
}

int main() {
    std::memset(&g_weights, 0, sizeof(g_weights));
    const bool real_data = init_real_data();
    if (!real_data) {
        std::cout << "Real Stage-1 data not found; using synthetic smoke data.\n";
        init_input();
        init_weights();
        init_reorder_tables();
        init_kernel_tables();
    }

    ifan_stage1_top(
        g_input,
        g_weights,
        g_reorder_r2_stem,
        g_reorder_r2_main,
        g_reorder_r1,
        g_kernel_idx_stem,
        g_kernel_idx_main,
        g_output
    );

    double checksum = 0.0;
    double abs_sum = 0.0;
    double sq_err_sum = 0.0;
    double max_abs_err = 0.0;
    double golden_abs_sum = 0.0;
    data_t worst_out = 0.0f;
    data_t worst_ref = 0.0f;
    int worst_t = 0;
    int worst_c = 0;
    int worst_r = 0;
    int worst_ch = 0;
    int worst_h = 0;
    int worst_w = 0;
    data_t min_v = g_output[0][0][0][0][0][0];
    data_t max_v = min_v;
    std::size_t count = 0;
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
            for (int r = 0; r < IFAN_R_FULL; r++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R1; h++) {
                        for (int w = 0; w < IFAN_W_R1; w++) {
                            const data_t v = g_output[t][c][r][ch][h][w];
                            if (!std::isfinite(v)) {
                                std::cerr << "FAIL: non-finite output detected\n";
                                return 1;
                            }
                            checksum += static_cast<double>(v);
                            abs_sum += std::fabs(static_cast<double>(v));
                            min_v = v < min_v ? v : min_v;
                            max_v = v > max_v ? v : max_v;
                            count++;
                            if (real_data) {
                                const data_t ref = g_golden[t][c][r][ch][h][w];
                                const double err = static_cast<double>(v) - static_cast<double>(ref);
                                const double abs_err = std::fabs(err);
                                sq_err_sum += err * err;
                                golden_abs_sum += std::fabs(static_cast<double>(ref));
                                if (abs_err > max_abs_err) {
                                    max_abs_err = abs_err;
                                    worst_out = v;
                                    worst_ref = ref;
                                    worst_t = t;
                                    worst_c = c;
                                    worst_r = r;
                                    worst_ch = ch;
                                    worst_h = h;
                                    worst_w = w;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "IFAN Stage-1 HLS smoke test\n";
    std::cout << "Output shape: [" << IFAN_STAGE1_T << ", "
              << IFAN_BRANCH_CHANNELS << ", " << IFAN_R_FULL << ", "
              << IFAN_CHARTS << ", " << IFAN_H_R1 << ", " << IFAN_W_R1 << "]\n";
    std::cout << "Checksum: " << checksum << "\n";
    std::cout << "AbsSum: " << abs_sum << "\n";
    std::cout << "Min/Max: " << min_v << " / " << max_v << "\n";
    if (real_data) {
        const double rmse = std::sqrt(sq_err_sum / static_cast<double>(count));
        const double mean_abs_ref = golden_abs_sum / static_cast<double>(count);
        std::cout << "Golden: final_head_logits.txt\n";
        std::cout << "MaxAbsError: " << max_abs_err << "\n";
        std::cout << "RMSE: " << rmse << "\n";
        std::cout << "MeanAbsGolden: " << mean_abs_ref << "\n";
        std::cout << "WorstIndex: [" << worst_t << ", " << worst_c << ", " << worst_r
                  << ", " << worst_ch << ", " << worst_h << ", " << worst_w << "]\n";
        std::cout << "WorstOut/Ref: " << worst_out << " / " << worst_ref << "\n";
        if (max_abs_err > 1.0e-4 || rmse > 1.0e-5) {
            std::cerr << "FAIL: real Stage-1 output exceeds tolerance\n";
            return 1;
        }
    }
    std::cout << "PASS\n";
    return 0;
}
