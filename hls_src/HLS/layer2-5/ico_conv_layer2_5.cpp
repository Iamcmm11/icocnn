#include "ico_conv_layer2_5.hpp"

void clean_vertices(
    data_t input[CHARTS][H][W],
    data_t output[CHARTS][H][W]
) {
    for (int c = 0; c < CHARTS; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = input[c][h][w];
            }
        }
    }
    for (int c = 0; c < CHARTS; c++) {
        output[c][0][0] = 0.0f;
        output[c][0][H] = 0.0f;
    }
}

void smooth_vertices(
    data_t input[CIN][RIN][CHARTS][H][W],
    data_t output[CIN][RIN][CHARTS][H][W]
) {
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        output[ci][ri][c][h][w] = input[ci][ri][c][h][w];
                    }
                }
            }
        }
    }

    for (int ci = 0; ci < CIN; ci++) {
        for (int c = 0; c < CHARTS; c++) {
            int prev_c = (c - 1 + CHARTS) % CHARTS;
            float sum_v1 = 0.0f;
            float sum_v2 = 0.0f;
            for (int ri = 0; ri < RIN; ri++) {
                sum_v1 += input[ci][ri][c][1][0];
                sum_v1 += input[ci][ri][c][1][1];
                sum_v1 += input[ci][ri][c][0][1];
                sum_v1 += input[ci][ri][prev_c][H - 1][H];
                sum_v1 += input[ci][ri][prev_c][H - 1][H - 1];

                sum_v2 += input[ci][ri][c][1][H];
                sum_v2 += input[ci][ri][c][1][(H + 1) % W];
                sum_v2 += input[ci][ri][c][0][(H + 1) % W];
                sum_v2 += input[ci][ri][prev_c][H - 1][W - 1];
                sum_v2 += input[ci][ri][c][0][H - 1];
            }
            float mean_v1 = sum_v1 / (RIN * 5.0f);
            float mean_v2 = sum_v2 / (RIN * 5.0f);
            for (int ri = 0; ri < RIN; ri++) {
                output[ci][ri][c][0][0] = mean_v1;
                output[ci][ri][c][0][H] = mean_v2;
            }
        }
    }
}

void pad_ico(
    data_t input[CIN][RIN][CHARTS][H][W],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[CIN][RIN][CHARTS][H_PADDED][W_PADDED]
) {
    data_t north_vertex[CIN][CHARTS];
    data_t south_vertex[CIN][CHARTS];

    for (int ci = 0; ci < CIN; ci++) {
        float smooth_north_pole_sum = 0.0f;
        float smooth_south_pole_sum = 0.0f;

        for (int c = 0; c < CHARTS; c++) {
            int prev_c = (c - 1 + CHARTS) % CHARTS;
            float sum_v1 = 0.0f;
            float sum_v2 = 0.0f;
            for (int ri = 0; ri < RIN; ri++) {
                sum_v1 += input[ci][ri][c][1][0];
                sum_v1 += input[ci][ri][c][1][1];
                sum_v1 += input[ci][ri][c][0][1];
                sum_v1 += input[ci][ri][prev_c][H - 1][H];
                sum_v1 += input[ci][ri][prev_c][H - 1][H - 1];

                sum_v2 += input[ci][ri][c][1][H];
                sum_v2 += input[ci][ri][c][1][(H + 1) % W];
                sum_v2 += input[ci][ri][c][0][(H + 1) % W];
                sum_v2 += input[ci][ri][prev_c][H - 1][W - 1];
                sum_v2 += input[ci][ri][c][0][H - 1];
            }
            north_vertex[ci][c] = sum_v1 / (RIN * 5.0f);
            south_vertex[ci][c] = sum_v2 / (RIN * 5.0f);
        }

        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                smooth_north_pole_sum += input[ci][ri][c][H - 1][0];
                smooth_south_pole_sum += input[ci][ri][c][0][W - 1];
            }
        }

        float smooth_north_pole = smooth_north_pole_sum / (RIN * CHARTS);
        float smooth_south_pole = smooth_south_pole_sum / (RIN * CHARTS);

        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                for (int h = 0; h < H_PADDED; h++) {
                    for (int w = 0; w < W_PADDED; w++) {
                        int reorder_val = reorder_idx[ri][c][h][w];
                        int src_ri = reorder_val / (CHARTS * H * W);
                        int remainder_ri = reorder_val % (CHARTS * H * W);
                        int src_chart = remainder_ri / (H * W);
                        int remainder = remainder_ri % (H * W);
                        int src_h = remainder / W;
                        int src_w = remainder % W;

                        data_t val = input[ci][src_ri][src_chart][src_h][src_w];
                        if (src_h == 0 && src_w == 0) {
                            val = north_vertex[ci][src_chart];
                        } else if (src_h == 0 && src_w == H) {
                            val = south_vertex[ci][src_chart];
                        }
                        output[ci][ri][c][h][w] = val;
                    }
                }
            }
        }

        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                output[ci][ri][c][H_PADDED - 1][1] = smooth_north_pole;
                output[ci][ri][c][1][W_PADDED - 1] = smooth_south_pole;
            }
        }
    }
}

static inline data_t expanded_weight_at(
    const data_t weight[COUT][CIN][RIN][7],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    int co,
    int ro,
    int ci,
    int ri,
    int kh,
    int kw
) {
    const int k = kh * KERNEL_W + kw;
    const int idx_cout = kernel_expansion_idx[co][ro][ci][ri][k][0];
    const int idx_cin = kernel_expansion_idx[co][ro][ci][ri][k][1];
    const int idx_rin = kernel_expansion_idx[co][ro][ci][ri][k][2];
    const int idx_w = kernel_expansion_idx[co][ro][ci][ri][k][3];

    if (idx_w < 0 || idx_w >= 7) {
        return 0.0f;
    }
    return weight[idx_cout][idx_cin][idx_rin][idx_w];
}

static inline void load_expanded_kernel(
    const data_t weight[COUT][CIN][RIN][7],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    int co,
    int ro,
    int ci,
    int ri,
    data_t kernel[KERNEL_H][KERNEL_W]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=0
    for (int kh = 0; kh < KERNEL_H; kh++) {
#pragma HLS UNROLL
        for (int kw = 0; kw < KERNEL_W; kw++) {
#pragma HLS UNROLL
            kernel[kh][kw] = expanded_weight_at(weight, kernel_expansion_idx, co, ro, ci, ri, kh, kw);
        }
    }
}

void conv_ico_layer2_5(
    data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W],
    const data_t weight[COUT][CIN][RIN][7],
    const data_t bias[COUT],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=4
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=OC_PAR_FACTOR dim=1

    for (int t = 0; t < TIME_STEPS; t++) {
        static data_t padded_frame[CIN][RIN][CHARTS][H_PADDED][W_PADDED];
#pragma HLS ARRAY_PARTITION variable=padded_frame complete dim=4
#pragma HLS ARRAY_PARTITION variable=padded_frame complete dim=5
        pad_ico(input[t], reorder_idx, padded_frame);

        for (int co = 0; co < COUT; co++) {
            data_t output_tile[ROUT][CHARTS][H][W];

            for (int ro = 0; ro < ROUT; ro++) {
                for (int sp = 0; sp < CHARTS * H * W; sp++) {
#pragma HLS PIPELINE II=1
                    const int c = sp / (H * W);
                    const int rem = sp % (H * W);
                    const int h = rem / W;
                    const int w = rem % W;
                    output_tile[ro][c][h][w] = bias[co];
                }
            }

            for (int ro = 0; ro < ROUT; ro++) {
                for (int ci = 0; ci < CIN; ci++) {
                    for (int ri = 0; ri < RIN; ri++) {
                        data_t kernel[KERNEL_H][KERNEL_W];
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=0
                        load_expanded_kernel(weight, kernel_expansion_idx, co, ro, ci, ri, kernel);

                        for (int sp = 0; sp < CHARTS * H * W; sp++) {
#pragma HLS PIPELINE II=1
                            const int c = sp / (H * W);
                            const int rem = sp % (H * W);
                            const int h = rem / W;
                            const int w = rem % W;

                            const data_t conv =
                                padded_frame[ci][ri][c][h + 0][w + 0] * kernel[0][0] +
                                padded_frame[ci][ri][c][h + 0][w + 1] * kernel[0][1] +
                                padded_frame[ci][ri][c][h + 0][w + 2] * kernel[0][2] +
                                padded_frame[ci][ri][c][h + 1][w + 0] * kernel[1][0] +
                                padded_frame[ci][ri][c][h + 1][w + 1] * kernel[1][1] +
                                padded_frame[ci][ri][c][h + 1][w + 2] * kernel[1][2] +
                                padded_frame[ci][ri][c][h + 2][w + 0] * kernel[2][0] +
                                padded_frame[ci][ri][c][h + 2][w + 1] * kernel[2][1] +
                                padded_frame[ci][ri][c][h + 2][w + 2] * kernel[2][2];

                            output_tile[ro][c][h][w] += conv;
                        }
                    }
                }
            }

            for (int ro = 0; ro < ROUT; ro++) {
                for (int c = 0; c < CHARTS; c++) {
                    output_tile[ro][c][0][0] = 0.0f;
                    output_tile[ro][c][0][H] = 0.0f;
                }
            }

            for (int c = 0; c < CHARTS; c++) {
                int prev_c = (c - 1 + CHARTS) % CHARTS;
                float sum_v1 = 0.0f;
                float sum_v2 = 0.0f;
                for (int ro = 0; ro < ROUT; ro++) {
                    sum_v1 += output_tile[ro][c][1][0];
                    sum_v1 += output_tile[ro][c][1][1];
                    sum_v1 += output_tile[ro][c][0][1];
                    sum_v1 += output_tile[ro][prev_c][H - 1][H];
                    sum_v1 += output_tile[ro][prev_c][H - 1][H - 1];
                    sum_v2 += output_tile[ro][c][1][H];
                    sum_v2 += output_tile[ro][c][1][(H + 1) % W];
                    sum_v2 += output_tile[ro][c][0][(H + 1) % W];
                    sum_v2 += output_tile[ro][prev_c][H - 1][W - 1];
                    sum_v2 += output_tile[ro][c][0][H - 1];
                }
                float mean_v1 = sum_v1 / (ROUT * 5.0f);
                float mean_v2 = sum_v2 / (ROUT * 5.0f);
                for (int ro = 0; ro < ROUT; ro++) {
                    output_tile[ro][c][0][0] = mean_v1;
                    output_tile[ro][c][0][H] = mean_v2;
                }
            }

            for (int ro = 0; ro < ROUT; ro++) {
                for (int sp = 0; sp < CHARTS * H * W; sp++) {
#pragma HLS PIPELINE II=1
                    const int c = sp / (H * W);
                    const int rem = sp % (H * W);
                    const int h = rem / W;
                    const int w = rem % W;
                    output[t][co][ro][c][h][w] = output_tile[ro][c][h][w];
                }
            }
        }
    }
}
