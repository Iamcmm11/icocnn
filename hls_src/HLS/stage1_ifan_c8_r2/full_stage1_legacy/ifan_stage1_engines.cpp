#include "ifan_stage1_engines.hpp"

static inline input_t to_input_t(data_t x) {
    return static_cast<input_t>(x);
}

static inline weight_t to_weight_t(data_t x) {
    return static_cast<weight_t>(x);
}

static inline act_t to_act_t(acc_t x) {
    return static_cast<act_t>(x);
}

static inline data_t to_data_t(acc_t x) {
    return static_cast<data_t>(x);
}

static inline data_t relu_scalar(data_t x) {
    return x > 0.0f ? x : 0.0f;
}

static inline data_t sigmoid_scalar(data_t x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static void init_output_tile_r2(
    const data_t bias[IFAN_BRANCH_CHANNELS],
    int co_base,
    act_t output_tile[IFAN_OC_TILE][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
        const act_t bias_val = static_cast<act_t>(bias[co_base + coo]);
        for (int ro = 0; ro < IFAN_R_FULL; ro++) {
            for (int sp = 0; sp < IFAN_CHARTS * IFAN_H_R2 * IFAN_W_R2; sp++) {
#pragma HLS PIPELINE II=1
                const int ch = sp / (IFAN_H_R2 * IFAN_W_R2);
                const int rem = sp % (IFAN_H_R2 * IFAN_W_R2);
                const int h = rem / IFAN_W_R2;
                const int w = rem % IFAN_W_R2;
                output_tile[coo][ro][ch][h][w] = bias_val;
            }
        }
    }
}

static void init_output_tile_r1(
    const data_t bias[IFAN_BRANCH_CHANNELS],
    int co_base,
    act_t output_tile[IFAN_OC_TILE][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
        const act_t bias_val = static_cast<act_t>(bias[co_base + coo]);
        for (int ro = 0; ro < IFAN_R_FULL; ro++) {
            for (int sp = 0; sp < IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1; sp++) {
#pragma HLS PIPELINE II=1
                const int ch = sp / (IFAN_H_R1 * IFAN_W_R1);
                const int rem = sp % (IFAN_H_R1 * IFAN_W_R1);
                const int h = rem / IFAN_W_R1;
                const int w = rem % IFAN_W_R1;
                output_tile[coo][ro][ch][h][w] = bias_val;
            }
        }
    }
}

static void writeback_output_tile_r2(
    act_t output_tile[IFAN_OC_TILE][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    int co_base,
    data_t output[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
        const int co = co_base + coo;
        for (int ro = 0; ro < IFAN_R_FULL; ro++) {
            for (int sp = 0; sp < IFAN_CHARTS * IFAN_H_R2 * IFAN_W_R2; sp++) {
#pragma HLS PIPELINE II=1
                const int ch = sp / (IFAN_H_R2 * IFAN_W_R2);
                const int rem = sp % (IFAN_H_R2 * IFAN_W_R2);
                const int h = rem / IFAN_W_R2;
                const int w = rem % IFAN_W_R2;
                output[co][ro][ch][h][w] = to_data_t(static_cast<acc_t>(output_tile[coo][ro][ch][h][w]));
            }
        }
    }
}

static void writeback_output_tile_r1(
    act_t output_tile[IFAN_OC_TILE][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    int co_base,
    data_t output[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
        const int co = co_base + coo;
        for (int ro = 0; ro < IFAN_R_FULL; ro++) {
            for (int sp = 0; sp < IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1; sp++) {
#pragma HLS PIPELINE II=1
                const int ch = sp / (IFAN_H_R1 * IFAN_W_R1);
                const int rem = sp % (IFAN_H_R1 * IFAN_W_R1);
                const int h = rem / IFAN_W_R1;
                const int w = rem % IFAN_W_R1;
                output[co][ro][ch][h][w] = to_data_t(static_cast<acc_t>(output_tile[coo][ro][ch][h][w]));
            }
        }
    }
}

static void stage_ico_stem_weight_tile(
    const data_t weight[IFAN_BRANCH_CHANNELS][1][1][IFAN_KERNEL_NEIGHBORS],
    const int kernel_idx[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][1][1][9][4],
    int co_base,
    int ro,
    weight_t kernel_tile[IFAN_OC_TILE][IFAN_KERNEL_H][IFAN_KERNEL_W]
) {
#pragma HLS ARRAY_PARTITION variable=kernel_tile complete dim=0
    for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
#pragma HLS UNROLL
        const int co = co_base + coo;
        for (int kh = 0; kh < IFAN_KERNEL_H; kh++) {
#pragma HLS UNROLL
            for (int kw = 0; kw < IFAN_KERNEL_W; kw++) {
#pragma HLS UNROLL
                const int k = kh * IFAN_KERNEL_W + kw;
                const int idx_co = kernel_idx[co][ro][0][0][k][0];
                const int idx_ci = kernel_idx[co][ro][0][0][k][1];
                const int idx_ri = kernel_idx[co][ro][0][0][k][2];
                const int idx_w = kernel_idx[co][ro][0][0][k][3];
                kernel_tile[coo][kh][kw] =
                    (idx_w >= 0 && idx_w < IFAN_KERNEL_NEIGHBORS)
                        ? to_weight_t(weight[idx_co][idx_ci][idx_ri][idx_w])
                        : static_cast<weight_t>(0);
            }
        }
    }
}

static void stage_ico_main_weight_tile(
    const data_t weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS],
    const int kernel_idx[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4],
    int co_base,
    int ro,
    weight_t staged_weight[IFAN_OC_TILE][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_H * IFAN_KERNEL_W],
    bool staged_valid[IFAN_OC_TILE][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_H * IFAN_KERNEL_W]
) {
    for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
#pragma HLS UNROLL
        const int co = co_base + coo;
        for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
            for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                for (int k = 0; k < IFAN_KERNEL_H * IFAN_KERNEL_W; k++) {
#pragma HLS PIPELINE II=1
                    const int idx_co = kernel_idx[co][ro][ci][ri][k][0];
                    const int idx_ci = kernel_idx[co][ro][ci][ri][k][1];
                    const int idx_ri = kernel_idx[co][ro][ci][ri][k][2];
                    const int idx_w = kernel_idx[co][ro][ci][ri][k][3];
                    const bool valid = idx_w >= 0 && idx_w < IFAN_KERNEL_NEIGHBORS;
                    staged_valid[coo][ci][ri][k] = valid;
                    staged_weight[coo][ci][ri][k] =
                        valid ? to_weight_t(weight[idx_co][idx_ci][idx_ri][idx_w]) : static_cast<weight_t>(0);
                }
            }
        }
    }
}

static void stage_temporal_weight_tile(
    const data_t weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL],
    int co_base,
    weight_t staged_weight[IFAN_OC_TILE][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL]
) {
    for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
#pragma HLS UNROLL
        const int co = co_base + coo;
        for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
            for (int k = 0; k < IFAN_TEMPORAL_KERNEL; k++) {
#pragma HLS PIPELINE II=1
                staged_weight[coo][ci][k] = to_weight_t(weight[co][ci][k]);
            }
        }
    }
}

static inline int decode_src_ri(int reorder_val, int height, int width) {
    return reorder_val / (IFAN_CHARTS * height * width);
}

static inline int decode_src_chart(int reorder_val, int height, int width) {
    const int rem_ri = reorder_val % (IFAN_CHARTS * height * width);
    return rem_ri / (height * width);
}

static inline int decode_src_h(int reorder_val, int height, int width) {
    const int rem_ri = reorder_val % (IFAN_CHARTS * height * width);
    const int rem_chart = rem_ri % (height * width);
    return rem_chart / width;
}

static inline int decode_src_w(int reorder_val, int height, int width) {
    const int rem_ri = reorder_val % (IFAN_CHARTS * height * width);
    const int rem_chart = rem_ri % (height * width);
    return rem_chart % width;
}

static inline data_t vertex_north_r2(
    input_t input[1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    int ch
) {
    const int prev_ch = (ch - 1 + IFAN_CHARTS) % IFAN_CHARTS;
    acc_t sum = 0;
    sum += static_cast<acc_t>(input[0][0][ch][1][0]);
    sum += static_cast<acc_t>(input[0][0][ch][1][1]);
    sum += static_cast<acc_t>(input[0][0][ch][0][1]);
    sum += static_cast<acc_t>(input[0][0][prev_ch][IFAN_H_R2 - 1][IFAN_H_R2]);
    sum += static_cast<acc_t>(input[0][0][prev_ch][IFAN_H_R2 - 1][IFAN_H_R2 - 1]);
    return to_data_t(sum / static_cast<acc_t>(5));
}

static inline data_t vertex_south_r2(
    input_t input[1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    int ch
) {
    const int prev_ch = (ch - 1 + IFAN_CHARTS) % IFAN_CHARTS;
    acc_t sum = 0;
    sum += static_cast<acc_t>(input[0][0][ch][1][IFAN_H_R2]);
    sum += static_cast<acc_t>(input[0][0][ch][1][IFAN_H_R2 + 1]);
    sum += static_cast<acc_t>(input[0][0][ch][0][IFAN_H_R2 + 1]);
    sum += static_cast<acc_t>(input[0][0][prev_ch][IFAN_H_R2 - 1][IFAN_W_R2 - 1]);
    sum += static_cast<acc_t>(input[0][0][ch][0][IFAN_H_R2 - 1]);
    return to_data_t(sum / static_cast<acc_t>(5));
}

static inline data_t vertex_north_r2_main(
    input_t input[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    int ci,
    int ch
) {
    const int prev_ch = (ch - 1 + IFAN_CHARTS) % IFAN_CHARTS;
    acc_t sum = 0;
    for (int ri = 0; ri < IFAN_R_FULL; ri++) {
        sum += static_cast<acc_t>(input[ci][ri][ch][1][0]);
        sum += static_cast<acc_t>(input[ci][ri][ch][1][1]);
        sum += static_cast<acc_t>(input[ci][ri][ch][0][1]);
        sum += static_cast<acc_t>(input[ci][ri][prev_ch][IFAN_H_R2 - 1][IFAN_H_R2]);
        sum += static_cast<acc_t>(input[ci][ri][prev_ch][IFAN_H_R2 - 1][IFAN_H_R2 - 1]);
    }
    return to_data_t(sum / static_cast<acc_t>(IFAN_R_FULL * 5));
}

static inline data_t vertex_south_r2_main(
    input_t input[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    int ci,
    int ch
) {
    const int prev_ch = (ch - 1 + IFAN_CHARTS) % IFAN_CHARTS;
    acc_t sum = 0;
    for (int ri = 0; ri < IFAN_R_FULL; ri++) {
        sum += static_cast<acc_t>(input[ci][ri][ch][1][IFAN_H_R2]);
        sum += static_cast<acc_t>(input[ci][ri][ch][1][IFAN_H_R2 + 1]);
        sum += static_cast<acc_t>(input[ci][ri][ch][0][IFAN_H_R2 + 1]);
        sum += static_cast<acc_t>(input[ci][ri][prev_ch][IFAN_H_R2 - 1][IFAN_W_R2 - 1]);
        sum += static_cast<acc_t>(input[ci][ri][ch][0][IFAN_H_R2 - 1]);
    }
    return to_data_t(sum / static_cast<acc_t>(IFAN_R_FULL * 5));
}

static inline data_t vertex_north_r1_main(
    input_t input[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    int ci,
    int ch
) {
    const int prev_ch = (ch - 1 + IFAN_CHARTS) % IFAN_CHARTS;
    acc_t sum = 0;
    for (int ri = 0; ri < IFAN_R_FULL; ri++) {
        sum += static_cast<acc_t>(input[ci][ri][ch][1][0]);
        sum += static_cast<acc_t>(input[ci][ri][ch][1][1]);
        sum += static_cast<acc_t>(input[ci][ri][ch][0][1]);
        sum += static_cast<acc_t>(input[ci][ri][prev_ch][IFAN_H_R1 - 1][IFAN_H_R1]);
        sum += static_cast<acc_t>(input[ci][ri][prev_ch][IFAN_H_R1 - 1][IFAN_H_R1 - 1]);
    }
    return to_data_t(sum / static_cast<acc_t>(IFAN_R_FULL * 5));
}

static inline data_t vertex_south_r1_main(
    input_t input[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    int ci,
    int ch
) {
    const int prev_ch = (ch - 1 + IFAN_CHARTS) % IFAN_CHARTS;
    acc_t sum = 0;
    for (int ri = 0; ri < IFAN_R_FULL; ri++) {
        sum += static_cast<acc_t>(input[ci][ri][ch][1][IFAN_H_R1]);
        sum += static_cast<acc_t>(input[ci][ri][ch][1][IFAN_H_R1 + 1]);
        sum += static_cast<acc_t>(input[ci][ri][ch][0][IFAN_H_R1 + 1]);
        sum += static_cast<acc_t>(input[ci][ri][prev_ch][IFAN_H_R1 - 1][IFAN_W_R1 - 1]);
        sum += static_cast<acc_t>(input[ci][ri][ch][0][IFAN_H_R1 - 1]);
    }
    return to_data_t(sum / static_cast<acc_t>(IFAN_R_FULL * 5));
}

static inline void smooth_poles_r2_stem(
    input_t input[1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    data_t north[IFAN_CHARTS],
    data_t south[IFAN_CHARTS],
    data_t &north_pole,
    data_t &south_pole
) {
    acc_t north_pole_sum = 0;
    acc_t south_pole_sum = 0;
    for (int ch = 0; ch < IFAN_CHARTS; ch++) {
        north[ch] = vertex_north_r2(input, ch);
        south[ch] = vertex_south_r2(input, ch);
        north_pole_sum += static_cast<acc_t>(input[0][0][ch][IFAN_H_R2 - 1][0]);
        south_pole_sum += static_cast<acc_t>(input[0][0][ch][0][IFAN_W_R2 - 1]);
    }
    north_pole = to_data_t(north_pole_sum / static_cast<acc_t>(IFAN_CHARTS));
    south_pole = to_data_t(south_pole_sum / static_cast<acc_t>(IFAN_CHARTS));
}

static inline void smooth_poles_r2_main(
    input_t input[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    int ci,
    data_t north[IFAN_CHARTS],
    data_t south[IFAN_CHARTS],
    data_t &north_pole,
    data_t &south_pole
) {
    acc_t north_pole_sum = 0;
    acc_t south_pole_sum = 0;
    for (int ch = 0; ch < IFAN_CHARTS; ch++) {
        north[ch] = vertex_north_r2_main(input, ci, ch);
        south[ch] = vertex_south_r2_main(input, ci, ch);
        for (int ri = 0; ri < IFAN_R_FULL; ri++) {
            north_pole_sum += static_cast<acc_t>(input[ci][ri][ch][IFAN_H_R2 - 1][0]);
            south_pole_sum += static_cast<acc_t>(input[ci][ri][ch][0][IFAN_W_R2 - 1]);
        }
    }
    north_pole = to_data_t(north_pole_sum / static_cast<acc_t>(IFAN_R_FULL * IFAN_CHARTS));
    south_pole = to_data_t(south_pole_sum / static_cast<acc_t>(IFAN_R_FULL * IFAN_CHARTS));
}

static inline void smooth_poles_r1_main(
    input_t input[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    int ci,
    data_t north[IFAN_CHARTS],
    data_t south[IFAN_CHARTS],
    data_t &north_pole,
    data_t &south_pole
) {
    acc_t north_pole_sum = 0;
    acc_t south_pole_sum = 0;
    for (int ch = 0; ch < IFAN_CHARTS; ch++) {
        north[ch] = vertex_north_r1_main(input, ci, ch);
        south[ch] = vertex_south_r1_main(input, ci, ch);
        for (int ri = 0; ri < IFAN_R_FULL; ri++) {
            north_pole_sum += static_cast<acc_t>(input[ci][ri][ch][IFAN_H_R1 - 1][0]);
            south_pole_sum += static_cast<acc_t>(input[ci][ri][ch][0][IFAN_W_R1 - 1]);
        }
    }
    north_pole = to_data_t(north_pole_sum / static_cast<acc_t>(IFAN_R_FULL * IFAN_CHARTS));
    south_pole = to_data_t(south_pole_sum / static_cast<acc_t>(IFAN_R_FULL * IFAN_CHARTS));
}

static void smooth_output_r2_frame(
    data_t x[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
        data_t north[IFAN_CHARTS];
        data_t south[IFAN_CHARTS];
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            const int prev_ch = (ch - 1 + IFAN_CHARTS) % IFAN_CHARTS;
            acc_t north_sum = 0;
            acc_t south_sum = 0;
            for (int ro = 0; ro < IFAN_R_FULL; ro++) {
                north_sum += static_cast<acc_t>(x[co][ro][ch][1][0]);
                north_sum += static_cast<acc_t>(x[co][ro][ch][1][1]);
                north_sum += static_cast<acc_t>(x[co][ro][ch][0][1]);
                north_sum += static_cast<acc_t>(x[co][ro][prev_ch][IFAN_H_R2 - 1][IFAN_H_R2]);
                north_sum += static_cast<acc_t>(x[co][ro][prev_ch][IFAN_H_R2 - 1][IFAN_H_R2 - 1]);

                south_sum += static_cast<acc_t>(x[co][ro][ch][1][IFAN_H_R2]);
                south_sum += static_cast<acc_t>(x[co][ro][ch][1][IFAN_H_R2 + 1]);
                south_sum += static_cast<acc_t>(x[co][ro][ch][0][IFAN_H_R2 + 1]);
                south_sum += static_cast<acc_t>(x[co][ro][prev_ch][IFAN_H_R2 - 1][IFAN_W_R2 - 1]);
                south_sum += static_cast<acc_t>(x[co][ro][ch][0][IFAN_H_R2 - 1]);
            }
            north[ch] = to_data_t(north_sum / static_cast<acc_t>(IFAN_R_FULL * 5));
            south[ch] = to_data_t(south_sum / static_cast<acc_t>(IFAN_R_FULL * 5));
        }
        for (int ro = 0; ro < IFAN_R_FULL; ro++) {
            for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                x[co][ro][ch][0][0] = north[ch];
                x[co][ro][ch][0][IFAN_H_R2] = south[ch];
            }
        }
    }
}

static void smooth_output_r1_frame(
    data_t x[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
        data_t north[IFAN_CHARTS];
        data_t south[IFAN_CHARTS];
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            const int prev_ch = (ch - 1 + IFAN_CHARTS) % IFAN_CHARTS;
            acc_t north_sum = 0;
            acc_t south_sum = 0;
            for (int ro = 0; ro < IFAN_R_FULL; ro++) {
                north_sum += static_cast<acc_t>(x[co][ro][ch][1][0]);
                north_sum += static_cast<acc_t>(x[co][ro][ch][1][1]);
                north_sum += static_cast<acc_t>(x[co][ro][ch][0][1]);
                north_sum += static_cast<acc_t>(x[co][ro][prev_ch][IFAN_H_R1 - 1][IFAN_H_R1]);
                north_sum += static_cast<acc_t>(x[co][ro][prev_ch][IFAN_H_R1 - 1][IFAN_H_R1 - 1]);

                south_sum += static_cast<acc_t>(x[co][ro][ch][1][IFAN_H_R1]);
                south_sum += static_cast<acc_t>(x[co][ro][ch][1][IFAN_H_R1 + 1]);
                south_sum += static_cast<acc_t>(x[co][ro][ch][0][IFAN_H_R1 + 1]);
                south_sum += static_cast<acc_t>(x[co][ro][prev_ch][IFAN_H_R1 - 1][IFAN_W_R1 - 1]);
                south_sum += static_cast<acc_t>(x[co][ro][ch][0][IFAN_H_R1 - 1]);
            }
            north[ch] = to_data_t(north_sum / static_cast<acc_t>(IFAN_R_FULL * 5));
            south[ch] = to_data_t(south_sum / static_cast<acc_t>(IFAN_R_FULL * 5));
        }
        for (int ro = 0; ro < IFAN_R_FULL; ro++) {
            for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                x[co][ro][ch][0][0] = north[ch];
                x[co][ro][ch][0][IFAN_H_R1] = south[ch];
            }
        }
    }
}

void relu_feature_r2(
    data_t x[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
            for (int r = 0; r < IFAN_R_FULL; r++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R2; h++) {
                        for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                            x[t][c][r][ch][h][w] = relu_scalar(x[t][c][r][ch][h][w]);
                        }
                    }
                }
            }
        }
    }
}

void relu_feature_r1(
    data_t x[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
            for (int r = 0; r < IFAN_R_FULL; r++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R1; h++) {
                        for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                            x[t][c][r][ch][h][w] = relu_scalar(x[t][c][r][ch][h][w]);
                        }
                    }
                }
            }
        }
    }
}

void sigmoid_feature_r2(
    data_t x[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
            for (int r = 0; r < IFAN_R_FULL; r++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R2; h++) {
                        for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                            x[t][c][r][ch][h][w] = sigmoid_scalar(x[t][c][r][ch][h][w]);
                        }
                    }
                }
            }
        }
    }
}

void attention_fuse_r2(
    data_t direct[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    data_t enhanced[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    data_t weight[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
            for (int r = 0; r < IFAN_R_FULL; r++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R2; h++) {
                        for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                            output[t][c][r][ch][h][w] =
                                direct[t][c][r][ch][h][w] +
                                enhanced[t][c][r][ch][h][w] * weight[t][c][r][ch][h][w];
                        }
                    }
                }
            }
        }
    }
}

void add_feature_r2(
    data_t a[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    data_t b[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
            for (int r = 0; r < IFAN_R_FULL; r++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R2; h++) {
                        for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                            output[t][c][r][ch][h][w] = a[t][c][r][ch][h][w] + b[t][c][r][ch][h][w];
                        }
                    }
                }
            }
        }
    }
}

static void pad_r2_stem_frame(
    input_t input[1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const int reorder_idx[1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    input_t padded[1][1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2]
) {
    data_t north[IFAN_CHARTS];
    data_t south[IFAN_CHARTS];
    data_t north_pole = 0.0f;
    data_t south_pole = 0.0f;
    smooth_poles_r2_stem(input, north, south, north_pole, south_pole);

    for (int ch = 0; ch < IFAN_CHARTS; ch++) {
        for (int h = 0; h < IFAN_H_R2 + 2; h++) {
            for (int w = 0; w < IFAN_W_R2 + 2; w++) {
#pragma HLS PIPELINE II=1
                const int rv = reorder_idx[0][ch][h][w];
                const int src_ch = decode_src_chart(rv, IFAN_H_R2, IFAN_W_R2);
                const int src_h = decode_src_h(rv, IFAN_H_R2, IFAN_W_R2);
                const int src_w = decode_src_w(rv, IFAN_H_R2, IFAN_W_R2);
                input_t val = input[0][0][src_ch][src_h][src_w];
                if (src_h == 0 && src_w == 0) {
                    val = to_input_t(north[src_ch]);
                } else if (src_h == 0 && src_w == IFAN_H_R2) {
                    val = to_input_t(south[src_ch]);
                }
                padded[0][0][ch][h][w] = val;
            }
        }
    }

    for (int ch = 0; ch < IFAN_CHARTS; ch++) {
        padded[0][0][ch][IFAN_H_R2 + 1][1] = to_input_t(north_pole);
        padded[0][0][ch][1][IFAN_W_R2 + 1] = to_input_t(south_pole);
    }
}

static void pad_r2_main_frame(
    input_t input[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const int reorder_idx[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    input_t padded[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2]
) {
    for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
        data_t north[IFAN_CHARTS];
        data_t south[IFAN_CHARTS];
        data_t north_pole = 0.0f;
        data_t south_pole = 0.0f;
        smooth_poles_r2_main(input, ci, north, south, north_pole, south_pole);

        for (int ri = 0; ri < IFAN_R_FULL; ri++) {
            for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                for (int h = 0; h < IFAN_H_R2 + 2; h++) {
                    for (int w = 0; w < IFAN_W_R2 + 2; w++) {
#pragma HLS PIPELINE II=1
                        const int rv = reorder_idx[ri][ch][h][w];
                        const int src_ri = decode_src_ri(rv, IFAN_H_R2, IFAN_W_R2);
                        const int src_ch = decode_src_chart(rv, IFAN_H_R2, IFAN_W_R2);
                        const int src_h = decode_src_h(rv, IFAN_H_R2, IFAN_W_R2);
                        const int src_w = decode_src_w(rv, IFAN_H_R2, IFAN_W_R2);
                        input_t val = input[ci][src_ri][src_ch][src_h][src_w];
                        if (src_h == 0 && src_w == 0) {
                            val = to_input_t(north[src_ch]);
                        } else if (src_h == 0 && src_w == IFAN_H_R2) {
                            val = to_input_t(south[src_ch]);
                        }
                        padded[ci][ri][ch][h][w] = val;
                    }
                }
            }
        }

        for (int ri = 0; ri < IFAN_R_FULL; ri++) {
            for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                padded[ci][ri][ch][IFAN_H_R2 + 1][1] = to_input_t(north_pole);
                padded[ci][ri][ch][1][IFAN_W_R2 + 1] = to_input_t(south_pole);
            }
        }
    }
}

static void pad_r1_main_frame(
    input_t input[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const int reorder_idx[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1 + 2][IFAN_W_R1 + 2],
    input_t padded[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1 + 2][IFAN_W_R1 + 2]
) {
    for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
        data_t north[IFAN_CHARTS];
        data_t south[IFAN_CHARTS];
        data_t north_pole = 0.0f;
        data_t south_pole = 0.0f;
        smooth_poles_r1_main(input, ci, north, south, north_pole, south_pole);

        for (int ri = 0; ri < IFAN_R_FULL; ri++) {
            for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                for (int h = 0; h < IFAN_H_R1 + 2; h++) {
                    for (int w = 0; w < IFAN_W_R1 + 2; w++) {
#pragma HLS PIPELINE II=1
                        const int rv = reorder_idx[ri][ch][h][w];
                        const int src_ri = decode_src_ri(rv, IFAN_H_R1, IFAN_W_R1);
                        const int src_ch = decode_src_chart(rv, IFAN_H_R1, IFAN_W_R1);
                        const int src_h = decode_src_h(rv, IFAN_H_R1, IFAN_W_R1);
                        const int src_w = decode_src_w(rv, IFAN_H_R1, IFAN_W_R1);
                        input_t val = input[ci][src_ri][src_ch][src_h][src_w];
                        if (src_h == 0 && src_w == 0) {
                            val = to_input_t(north[src_ch]);
                        } else if (src_h == 0 && src_w == IFAN_H_R1) {
                            val = to_input_t(south[src_ch]);
                        }
                        padded[ci][ri][ch][h][w] = val;
                    }
                }
            }
        }

        for (int ri = 0; ri < IFAN_R_FULL; ri++) {
            for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                padded[ci][ri][ch][IFAN_H_R1 + 1][1] = to_input_t(north_pole);
                padded[ci][ri][ch][1][IFAN_W_R1 + 1] = to_input_t(south_pole);
            }
        }
    }
}

void ico_conv_r2_stem_engine(
    data_t input[IFAN_STAGE1_T][1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const data_t weight[IFAN_BRANCH_CHANNELS][1][1][IFAN_KERNEL_NEIGHBORS],
    const data_t bias[IFAN_BRANCH_CHANNELS],
    const int kernel_idx[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][1][1][9][4],
    const int reorder_idx[1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=IFAN_OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=4
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=IFAN_OC_PAR_FACTOR dim=1

    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        static input_t staged[1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
        static input_t padded[1][1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2];
#pragma HLS ARRAY_PARTITION variable=staged complete dim=4
#pragma HLS ARRAY_PARTITION variable=staged complete dim=5
#pragma HLS ARRAY_PARTITION variable=padded complete dim=4
#pragma HLS ARRAY_PARTITION variable=padded complete dim=5

        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R2; h++) {
                for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                    staged[0][0][ch][h][w] = to_input_t(input[t][0][0][ch][h][w]);
                }
            }
        }
        pad_r2_stem_frame(staged, reorder_idx, padded);

        for (int co_base = 0; co_base < IFAN_BRANCH_CHANNELS; co_base += IFAN_OC_TILE) {
            act_t output_tile[IFAN_OC_TILE][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=4
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=5
            init_output_tile_r2(bias, co_base, output_tile);

            for (int ro = 0; ro < IFAN_R_FULL; ro++) {
                weight_t kernel_tile[IFAN_OC_TILE][IFAN_KERNEL_H][IFAN_KERNEL_W];
#pragma HLS ARRAY_PARTITION variable=kernel_tile complete dim=0
#pragma HLS ARRAY_PARTITION variable=kernel_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=kernel_tile complete dim=3
                stage_ico_stem_weight_tile(weight, kernel_idx, co_base, ro, kernel_tile);

                for (int sp = 0; sp < IFAN_CHARTS * IFAN_H_R2 * IFAN_W_R2; sp++) {
#pragma HLS PIPELINE II=1
                    const int ch = sp / (IFAN_H_R2 * IFAN_W_R2);
                    const int rem = sp % (IFAN_H_R2 * IFAN_W_R2);
                    const int h = rem / IFAN_W_R2;
                    const int w = rem % IFAN_W_R2;

                    for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
#pragma HLS UNROLL
                        const acc_t conv =
                            static_cast<acc_t>(padded[0][0][ch][h + 0][w + 0]) * static_cast<acc_t>(kernel_tile[coo][0][0]) +
                            static_cast<acc_t>(padded[0][0][ch][h + 0][w + 1]) * static_cast<acc_t>(kernel_tile[coo][0][1]) +
                            static_cast<acc_t>(padded[0][0][ch][h + 0][w + 2]) * static_cast<acc_t>(kernel_tile[coo][0][2]) +
                            static_cast<acc_t>(padded[0][0][ch][h + 1][w + 0]) * static_cast<acc_t>(kernel_tile[coo][1][0]) +
                            static_cast<acc_t>(padded[0][0][ch][h + 1][w + 1]) * static_cast<acc_t>(kernel_tile[coo][1][1]) +
                            static_cast<acc_t>(padded[0][0][ch][h + 1][w + 2]) * static_cast<acc_t>(kernel_tile[coo][1][2]) +
                            static_cast<acc_t>(padded[0][0][ch][h + 2][w + 0]) * static_cast<acc_t>(kernel_tile[coo][2][0]) +
                            static_cast<acc_t>(padded[0][0][ch][h + 2][w + 1]) * static_cast<acc_t>(kernel_tile[coo][2][1]) +
                            static_cast<acc_t>(padded[0][0][ch][h + 2][w + 2]) * static_cast<acc_t>(kernel_tile[coo][2][2]);
                        output_tile[coo][ro][ch][h][w] = static_cast<act_t>(
                            static_cast<acc_t>(output_tile[coo][ro][ch][h][w]) + conv
                        );
                    }
                }
            }
            writeback_output_tile_r2(output_tile, co_base, output[t]);
        }
        smooth_output_r2_frame(output[t]);
    }
}

void ico_conv_r2_main_engine(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const data_t weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS],
    const data_t bias[IFAN_BRANCH_CHANNELS],
    const int kernel_idx[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4],
    const int reorder_idx[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=IFAN_OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=4
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=IFAN_OC_PAR_FACTOR dim=1

    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        static input_t staged[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
        static input_t padded[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2];
#pragma HLS ARRAY_PARTITION variable=staged complete dim=4
#pragma HLS ARRAY_PARTITION variable=staged complete dim=5
#pragma HLS ARRAY_PARTITION variable=padded complete dim=4
#pragma HLS ARRAY_PARTITION variable=padded complete dim=5

        for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
            for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R2; h++) {
                        for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                            staged[ci][ri][ch][h][w] = to_input_t(input[t][ci][ri][ch][h][w]);
                        }
                    }
                }
            }
        }
        pad_r2_main_frame(staged, reorder_idx, padded);

        for (int co_base = 0; co_base < IFAN_BRANCH_CHANNELS; co_base += IFAN_OC_TILE) {
            act_t output_tile[IFAN_OC_TILE][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=4
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=5
            init_output_tile_r2(bias, co_base, output_tile);

            for (int ro = 0; ro < IFAN_R_FULL; ro++) {
                for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
                    acc_t ri_partial[IFAN_OC_TILE][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=1
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=2
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=4
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=5

                    weight_t staged_weight[IFAN_OC_TILE][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_H * IFAN_KERNEL_W];
                    bool staged_valid[IFAN_OC_TILE][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_H * IFAN_KERNEL_W];
#pragma HLS ARRAY_PARTITION variable=staged_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=staged_weight complete dim=4
#pragma HLS ARRAY_PARTITION variable=staged_valid complete dim=1
#pragma HLS ARRAY_PARTITION variable=staged_valid complete dim=4
                    stage_ico_main_weight_tile(weight, kernel_idx, co_base, ro, staged_weight, staged_valid);

                    for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                        for (int sp = 0; sp < IFAN_CHARTS * IFAN_H_R2 * IFAN_W_R2; sp++) {
#pragma HLS PIPELINE II=1
                            const int ch = sp / (IFAN_H_R2 * IFAN_W_R2);
                            const int rem = sp % (IFAN_H_R2 * IFAN_W_R2);
                            const int h = rem / IFAN_W_R2;
                            const int w = rem % IFAN_W_R2;

                            for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
#pragma HLS UNROLL
                                acc_t conv = 0;
                                for (int kh = 0; kh < IFAN_KERNEL_H; kh++) {
#pragma HLS UNROLL
                                    for (int kw = 0; kw < IFAN_KERNEL_W; kw++) {
#pragma HLS UNROLL
                                        const int k = kh * IFAN_KERNEL_W + kw;
                                        if (staged_valid[coo][ci][ri][k]) {
                                            conv += static_cast<acc_t>(padded[ci][ri][ch][h + kh][w + kw]) *
                                                    static_cast<acc_t>(staged_weight[coo][ci][ri][k]);
                                        }
                                    }
                                }
                                ri_partial[coo][ri][ch][h][w] = conv;
                            }
                        }
                    }

                    for (int sp = 0; sp < IFAN_CHARTS * IFAN_H_R2 * IFAN_W_R2; sp++) {
#pragma HLS PIPELINE II=1
                        const int ch = sp / (IFAN_H_R2 * IFAN_W_R2);
                        const int rem = sp % (IFAN_H_R2 * IFAN_W_R2);
                        const int h = rem / IFAN_W_R2;
                        const int w = rem % IFAN_W_R2;

                        for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
#pragma HLS UNROLL
                            acc_t ri_sum = 0;
                            for (int ri = 0; ri < IFAN_R_FULL; ri++) {
#pragma HLS UNROLL
                                ri_sum += ri_partial[coo][ri][ch][h][w];
                            }
                            output_tile[coo][ro][ch][h][w] = static_cast<act_t>(
                                static_cast<acc_t>(output_tile[coo][ro][ch][h][w]) + ri_sum
                            );
                        }
                    }
                }
            }
            writeback_output_tile_r2(output_tile, co_base, output[t]);
        }
        smooth_output_r2_frame(output[t]);
    }
}

void ico_conv_r1_main_engine(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS],
    const data_t bias[IFAN_BRANCH_CHANNELS],
    const int kernel_idx[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4],
    const int reorder_idx[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1 + 2][IFAN_W_R1 + 2],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=IFAN_OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=4
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=IFAN_OC_PAR_FACTOR dim=1

    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        static input_t staged[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
        static input_t padded[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1 + 2][IFAN_W_R1 + 2];
#pragma HLS ARRAY_PARTITION variable=staged complete dim=4
#pragma HLS ARRAY_PARTITION variable=staged complete dim=5
#pragma HLS ARRAY_PARTITION variable=padded complete dim=4
#pragma HLS ARRAY_PARTITION variable=padded complete dim=5

        for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
            for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R1; h++) {
                        for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                            staged[ci][ri][ch][h][w] = to_input_t(input[t][ci][ri][ch][h][w]);
                        }
                    }
                }
            }
        }
        pad_r1_main_frame(staged, reorder_idx, padded);

        for (int co_base = 0; co_base < IFAN_BRANCH_CHANNELS; co_base += IFAN_OC_TILE) {
            act_t output_tile[IFAN_OC_TILE][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=4
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=5
            init_output_tile_r1(bias, co_base, output_tile);

            for (int ro = 0; ro < IFAN_R_FULL; ro++) {
                for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
                    acc_t ri_partial[IFAN_OC_TILE][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=1
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=2
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=4
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=5

                    weight_t staged_weight[IFAN_OC_TILE][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_H * IFAN_KERNEL_W];
                    bool staged_valid[IFAN_OC_TILE][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_H * IFAN_KERNEL_W];
#pragma HLS ARRAY_PARTITION variable=staged_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=staged_weight complete dim=4
#pragma HLS ARRAY_PARTITION variable=staged_valid complete dim=1
#pragma HLS ARRAY_PARTITION variable=staged_valid complete dim=4
                    stage_ico_main_weight_tile(weight, kernel_idx, co_base, ro, staged_weight, staged_valid);

                    for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                        for (int sp = 0; sp < IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1; sp++) {
#pragma HLS PIPELINE II=1
                            const int ch = sp / (IFAN_H_R1 * IFAN_W_R1);
                            const int rem = sp % (IFAN_H_R1 * IFAN_W_R1);
                            const int h = rem / IFAN_W_R1;
                            const int w = rem % IFAN_W_R1;

                            for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
#pragma HLS UNROLL
                                acc_t conv = 0;
                                for (int kh = 0; kh < IFAN_KERNEL_H; kh++) {
#pragma HLS UNROLL
                                    for (int kw = 0; kw < IFAN_KERNEL_W; kw++) {
#pragma HLS UNROLL
                                        const int k = kh * IFAN_KERNEL_W + kw;
                                        if (staged_valid[coo][ci][ri][k]) {
                                            conv += static_cast<acc_t>(padded[ci][ri][ch][h + kh][w + kw]) *
                                                    static_cast<acc_t>(staged_weight[coo][ci][ri][k]);
                                        }
                                    }
                                }
                                ri_partial[coo][ri][ch][h][w] = conv;
                            }
                        }
                    }

                    for (int sp = 0; sp < IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1; sp++) {
#pragma HLS PIPELINE II=1
                        const int ch = sp / (IFAN_H_R1 * IFAN_W_R1);
                        const int rem = sp % (IFAN_H_R1 * IFAN_W_R1);
                        const int h = rem / IFAN_W_R1;
                        const int w = rem % IFAN_W_R1;

                        for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
#pragma HLS UNROLL
                            acc_t ri_sum = 0;
                            for (int ri = 0; ri < IFAN_R_FULL; ri++) {
#pragma HLS UNROLL
                                ri_sum += ri_partial[coo][ri][ch][h][w];
                            }
                            output_tile[coo][ro][ch][h][w] = static_cast<act_t>(
                                static_cast<acc_t>(output_tile[coo][ro][ch][h][w]) + ri_sum
                            );
                        }
                    }
                }
            }
            writeback_output_tile_r1(output_tile, co_base, output[t]);
        }
        smooth_output_r1_frame(output[t]);
    }
}

void pool_ico_r2_to_r1_engine(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const int reorder_idx[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        input_t staged[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
        input_t padded[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2];
#pragma HLS ARRAY_PARTITION variable=padded complete dim=4
#pragma HLS ARRAY_PARTITION variable=padded complete dim=5

        for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
            for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R2; h++) {
                        for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                            staged[ci][ri][ch][h][w] = to_input_t(input[t][ci][ri][ch][h][w]);
                        }
                    }
                }
            }
        }
        pad_r2_main_frame(staged, reorder_idx, padded);

        for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
            for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R1; h++) {
                        for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                            const int ph = 1 + 2 * h;
                            const int pw = 1 + 2 * w;
                            acc_t sum = 0;
                            sum += padded[ci][ri][ch][ph][pw];
                            sum += padded[ci][ri][ch][ph + 1][pw];
                            sum += padded[ci][ri][ch][ph + 1][pw + 1];
                            sum += padded[ci][ri][ch][ph][pw + 1];
                            sum += padded[ci][ri][ch][ph - 1][pw];
                            sum += padded[ci][ri][ch][ph - 1][pw - 1];
                            sum += padded[ci][ri][ch][ph][pw - 1];
                            output[t][ci][ri][ch][h][w] = to_data_t(sum / static_cast<acc_t>(7));
                        }
                    }
                }
            }
        }
        smooth_output_r1_frame(output[t]);
    }
}

void temporal_conv1d_r1_engine(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL],
    const data_t bias[IFAN_BRANCH_CHANNELS],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int ri = 0; ri < IFAN_R_FULL; ri++) {
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1; h++) {
                for (int w = 0; w < IFAN_W_R1; w++) {
                    for (int co_base = 0; co_base < IFAN_BRANCH_CHANNELS; co_base += IFAN_OC_TILE) {
                        weight_t staged_weight[IFAN_OC_TILE][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL];
#pragma HLS ARRAY_PARTITION variable=staged_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=staged_weight complete dim=3
                        stage_temporal_weight_tile(weight, co_base, staged_weight);

                        for (int t = 0; t < IFAN_STAGE1_T; t++) {
#pragma HLS PIPELINE II=1
                            for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
#pragma HLS UNROLL
                                acc_t sum = static_cast<acc_t>(bias[co_base + coo]);
                                for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
                                    for (int k = 0; k < IFAN_TEMPORAL_KERNEL; k++) {
#pragma HLS UNROLL
                                        const int src_t = t - (IFAN_TEMPORAL_KERNEL - 1) + k;
                                        if (src_t >= 0) {
                                            sum += static_cast<acc_t>(input[src_t][ci][ri][ch][h][w]) *
                                                   static_cast<acc_t>(staged_weight[coo][ci][k]);
                                        }
                                    }
                                }
                                output[t][co_base + coo][ri][ch][h][w] = to_data_t(sum);
                            }
                        }
                    }
                }
            }
        }
    }
}

void lnorm_ico_r2_engine(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const data_t gamma[IFAN_BRANCH_CHANNELS],
    const data_t beta[IFAN_BRANCH_CHANNELS],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    const int n = IFAN_BRANCH_CHANNELS * IFAN_R_FULL;
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R2; h++) {
                for (int w = 0; w < IFAN_W_R2; w++) {
                    acc_t mean = 0;
                    for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
                        for (int r = 0; r < IFAN_R_FULL; r++) {
                            mean += static_cast<acc_t>(input[t][c][r][ch][h][w]);
                        }
                    }
                    mean /= static_cast<acc_t>(n);
                    acc_t var = 0;
                    for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
                        for (int r = 0; r < IFAN_R_FULL; r++) {
                            const acc_t diff = static_cast<acc_t>(input[t][c][r][ch][h][w]) - mean;
                            var += diff * diff;
                        }
                    }
                    var /= static_cast<acc_t>(n);
                    const data_t inv_std = 1.0f / std::sqrt(static_cast<data_t>(var) + 1.0e-5f);
                    for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
                        for (int r = 0; r < IFAN_R_FULL; r++) {
#pragma HLS PIPELINE II=1
                            const data_t normed = (input[t][c][r][ch][h][w] - static_cast<data_t>(mean)) * inv_std;
                            output[t][c][r][ch][h][w] = normed * gamma[c] + beta[c];
                        }
                    }
                }
            }
        }
    }
}

void lnorm_ico_r1_engine(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t gamma[IFAN_BRANCH_CHANNELS],
    const data_t beta[IFAN_BRANCH_CHANNELS],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    const int n = IFAN_BRANCH_CHANNELS * IFAN_R_FULL;
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1; h++) {
                for (int w = 0; w < IFAN_W_R1; w++) {
                    acc_t mean = 0;
                    for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
                        for (int r = 0; r < IFAN_R_FULL; r++) {
                            mean += static_cast<acc_t>(input[t][c][r][ch][h][w]);
                        }
                    }
                    mean /= static_cast<acc_t>(n);
                    acc_t var = 0;
                    for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
                        for (int r = 0; r < IFAN_R_FULL; r++) {
                            const acc_t diff = static_cast<acc_t>(input[t][c][r][ch][h][w]) - mean;
                            var += diff * diff;
                        }
                    }
                    var /= static_cast<acc_t>(n);
                    const data_t inv_std = 1.0f / std::sqrt(static_cast<data_t>(var) + 1.0e-5f);
                    for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
                        for (int r = 0; r < IFAN_R_FULL; r++) {
#pragma HLS PIPELINE II=1
                            const data_t normed = (input[t][c][r][ch][h][w] - static_cast<data_t>(mean)) * inv_std;
                            output[t][c][r][ch][h][w] = normed * gamma[c] + beta[c];
                        }
                    }
                }
            }
        }
    }
}
