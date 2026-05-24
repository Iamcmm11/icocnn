#include "ifan_stage1.hpp"

#include <cmath>
#include <cstring>
#include <iostream>

static data_t g_input[IFAN_IN_CHANNELS][IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
static IfanStage1Weights g_weights;
static int g_reorder_r2_stem[1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2];
static int g_reorder_r2_main[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2];
static int g_reorder_r1[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1 + 2][IFAN_W_R1 + 2];
static int g_kernel_idx_stem[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][1][1][9][4];
static int g_kernel_idx_main[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4];
static data_t g_output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];

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

int main() {
    init_input();
    init_weights();
    init_reorder_tables();
    init_kernel_tables();

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
    data_t min_v = g_output[0][0][0][0][0][0];
    data_t max_v = min_v;
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
    std::cout << "PASS\n";
    return 0;
}
