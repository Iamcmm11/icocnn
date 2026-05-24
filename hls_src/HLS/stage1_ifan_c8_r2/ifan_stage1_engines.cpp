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
    for (int ch = 0; ch < IFAN_CHARTS; ch++) {
        for (int h = 0; h < IFAN_H_R2 + 2; h++) {
            for (int w = 0; w < IFAN_W_R2 + 2; w++) {
#pragma HLS PIPELINE II=1
                const int rv = reorder_idx[0][ch][h][w];
                const int src_ch = decode_src_chart(rv, IFAN_H_R2, IFAN_W_R2);
                const int src_h = decode_src_h(rv, IFAN_H_R2, IFAN_W_R2);
                const int src_w = decode_src_w(rv, IFAN_H_R2, IFAN_W_R2);
                padded[0][0][ch][h][w] = input[0][0][src_ch][src_h][src_w];
            }
        }
    }
}

static void pad_r2_main_frame(
    input_t input[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const int reorder_idx[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    input_t padded[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2]
) {
    for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
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
                        padded[ci][ri][ch][h][w] = input[ci][src_ri][src_ch][src_h][src_w];
                    }
                }
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
                        padded[ci][ri][ch][h][w] = input[ci][src_ri][src_ch][src_h][src_w];
                    }
                }
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
                                    const int idx_w = kernel_idx[co][ro][0][0][k][3];
                                    if (idx_w >= 0 && idx_w < IFAN_KERNEL_NEIGHBORS) {
                                        sum += static_cast<acc_t>(padded[0][0][ch][h + kh][w + kw]) *
                                               static_cast<acc_t>(to_weight_t(weight[co][0][0][idx_w]));
                                    }
                                }
                            }
                            output[t][co][ro][ch][h][w] = to_data_t(sum);
                        }
                    }
                }
            }
        }
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
                                            const int idx_ci = kernel_idx[co][ro][ci][ri][k][1];
                                            const int idx_ri = kernel_idx[co][ro][ci][ri][k][2];
                                            const int idx_w = kernel_idx[co][ro][ci][ri][k][3];
                                            if (idx_w >= 0 && idx_w < IFAN_KERNEL_NEIGHBORS) {
                                                sum += static_cast<acc_t>(padded[idx_ci][idx_ri][ch][h + kh][w + kw]) *
                                                       static_cast<acc_t>(to_weight_t(weight[co][idx_ci][idx_ri][idx_w]));
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
                                            const int idx_ci = kernel_idx[co][ro][ci][ri][k][1];
                                            const int idx_ri = kernel_idx[co][ro][ci][ri][k][2];
                                            const int idx_w = kernel_idx[co][ro][ci][ri][k][3];
                                            if (idx_w >= 0 && idx_w < IFAN_KERNEL_NEIGHBORS) {
                                                sum += static_cast<acc_t>(padded[idx_ci][idx_ri][ch][h + kh][w + kw]) *
                                                       static_cast<acc_t>(to_weight_t(weight[co][idx_ci][idx_ri][idx_w]));
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
