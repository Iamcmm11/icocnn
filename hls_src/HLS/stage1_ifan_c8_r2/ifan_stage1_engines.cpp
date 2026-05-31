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
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        input_t staged[1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
        input_t padded[1][1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2];
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

        for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
            for (int ro = 0; ro < IFAN_R_FULL; ro++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R2; h++) {
                        for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                            acc_t sum = static_cast<acc_t>(bias[co]);
                            for (int kh = 0; kh < IFAN_KERNEL_H; kh++) {
                                for (int kw = 0; kw < IFAN_KERNEL_W; kw++) {
#pragma HLS UNROLL
                                    const int k = kh * IFAN_KERNEL_W + kw;
                                    const int idx_co = kernel_idx[co][ro][0][0][k][0];
                                    const int idx_ci = kernel_idx[co][ro][0][0][k][1];
                                    const int idx_ri = kernel_idx[co][ro][0][0][k][2];
                                    const int idx_w = kernel_idx[co][ro][0][0][k][3];
                                    if (idx_w >= 0 && idx_w < IFAN_KERNEL_NEIGHBORS) {
                                        sum += static_cast<acc_t>(padded[0][0][ch][h + kh][w + kw]) *
                                               static_cast<acc_t>(to_weight_t(weight[idx_co][idx_ci][idx_ri][idx_w]));
                                    }
                                }
                            }
                            output[t][co][ro][ch][h][w] = to_data_t(sum);
                        }
                    }
                }
            }
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
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        input_t staged[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
        input_t padded[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2];
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

        for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
            for (int ro = 0; ro < IFAN_R_FULL; ro++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R2; h++) {
                        for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                            acc_t sum = static_cast<acc_t>(bias[co]);
                            for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
                                for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                                    for (int kh = 0; kh < IFAN_KERNEL_H; kh++) {
                                        for (int kw = 0; kw < IFAN_KERNEL_W; kw++) {
                                            const int k = kh * IFAN_KERNEL_W + kw;
                                            const int idx_co = kernel_idx[co][ro][ci][ri][k][0];
                                            const int idx_ci = kernel_idx[co][ro][ci][ri][k][1];
                                            const int idx_ri = kernel_idx[co][ro][ci][ri][k][2];
                                            const int idx_w = kernel_idx[co][ro][ci][ri][k][3];
                                            if (idx_w >= 0 && idx_w < IFAN_KERNEL_NEIGHBORS) {
                                                sum += static_cast<acc_t>(padded[ci][ri][ch][h + kh][w + kw]) *
                                                       static_cast<acc_t>(to_weight_t(weight[idx_co][idx_ci][idx_ri][idx_w]));
                                            }
                                        }
                                    }
                                }
                            }
                            output[t][co][ro][ch][h][w] = to_data_t(sum);
                        }
                    }
                }
            }
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
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        input_t staged[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
        input_t padded[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1 + 2][IFAN_W_R1 + 2];
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

        for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
            for (int ro = 0; ro < IFAN_R_FULL; ro++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R1; h++) {
                        for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                            acc_t sum = static_cast<acc_t>(bias[co]);
                            for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
                                for (int ri = 0; ri < IFAN_R_FULL; ri++) {
                                    for (int kh = 0; kh < IFAN_KERNEL_H; kh++) {
                                        for (int kw = 0; kw < IFAN_KERNEL_W; kw++) {
                                            const int k = kh * IFAN_KERNEL_W + kw;
                                            const int idx_co = kernel_idx[co][ro][ci][ri][k][0];
                                            const int idx_ci = kernel_idx[co][ro][ci][ri][k][1];
                                            const int idx_ri = kernel_idx[co][ro][ci][ri][k][2];
                                            const int idx_w = kernel_idx[co][ro][ci][ri][k][3];
                                            if (idx_w >= 0 && idx_w < IFAN_KERNEL_NEIGHBORS) {
                                                sum += static_cast<acc_t>(padded[ci][ri][ch][h + kh][w + kw]) *
                                                       static_cast<acc_t>(to_weight_t(weight[idx_co][idx_ci][idx_ri][idx_w]));
                                            }
                                        }
                                    }
                                }
                            }
                            output[t][co][ro][ch][h][w] = to_data_t(sum);
                        }
                    }
                }
            }
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
                    for (int t = 0; t < IFAN_STAGE1_T; t++) {
                        for (int co = 0; co < IFAN_BRANCH_CHANNELS; co++) {
#pragma HLS PIPELINE II=1
                            acc_t sum = static_cast<acc_t>(bias[co]);
                            for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
                                for (int k = 0; k < IFAN_TEMPORAL_KERNEL; k++) {
                                    const int src_t = t - (IFAN_TEMPORAL_KERNEL - 1) + k;
                                    if (src_t >= 0) {
                                        sum += static_cast<acc_t>(input[src_t][ci][ri][ch][h][w]) *
                                               static_cast<acc_t>(to_weight_t(weight[co][ci][k]));
                                    }
                                }
                            }
                            output[t][co][ri][ch][h][w] = to_data_t(sum);
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
