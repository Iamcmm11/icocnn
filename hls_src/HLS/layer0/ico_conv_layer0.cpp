#include "ico_conv_layer0.hpp"
#include <cstring>

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
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                output[ci][ri][c][0][0] = 0.0f;
                output[ci][ri][c][0][H] = 0.0f;
            }
        }
    }

    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                const int prev_c = (c - 1 + CHARTS) % CHARTS;

                float sum_v1 = 0.0f;
                sum_v1 += input[ci][ri][c][1][0];
                sum_v1 += input[ci][ri][c][1][1];
                sum_v1 += input[ci][ri][c][0][1];
                sum_v1 += input[ci][ri][prev_c][H - 1][H];
                sum_v1 += input[ci][ri][prev_c][H - 1][H - 1];
                output[ci][ri][c][0][0] = sum_v1 / 5.0f;

                float sum_v2 = 0.0f;
                sum_v2 += input[ci][ri][c][1][H];
                sum_v2 += input[ci][ri][c][1][(H + 1) % W];
                sum_v2 += input[ci][ri][c][0][(H + 1) % W];
                sum_v2 += input[ci][ri][prev_c][H - 1][W - 1];
                sum_v2 += input[ci][ri][c][0][H - 1];
                output[ci][ri][c][0][H] = sum_v2 / 5.0f;
            }
        }
    }
}

static void smooth_vertices_fast(
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
        const int prev_c = (c - 1 + CHARTS) % CHARTS;
        output[c][0][0] =
            (input[c][1][0] +
             input[c][1][1] +
             input[c][0][1] +
             input[prev_c][H - 1][H] +
             input[prev_c][H - 1][H - 1]) / 5.0f;

        output[c][0][H] =
            (input[c][1][H] +
             input[c][1][(H + 1) % W] +
             input[c][0][(H + 1) % W] +
             input[prev_c][H - 1][W - 1] +
             input[c][0][H - 1]) / 5.0f;
    }
}

void pad_ico(
    data_t input[CIN][RIN][CHARTS][H][W],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[RIN][CHARTS][H_PADDED][W_PADDED]
) {
    static_assert(CIN == 1, "layer0 fast pad assumes CIN == 1.");
    static_assert(RIN == 1, "layer0 fast pad assumes RIN == 1.");

    data_t smoothed_input[CHARTS][H][W];
    smooth_vertices_fast(input[0][0], smoothed_input);

    float smooth_north_pole_sum = 0.0f;
    float smooth_south_pole_sum = 0.0f;
    for (int c = 0; c < CHARTS; c++) {
        smooth_north_pole_sum += smoothed_input[c][H - 1][0];
        smooth_south_pole_sum += smoothed_input[c][0][W - 1];
    }
    const float smooth_north_pole = smooth_north_pole_sum / CHARTS;
    const float smooth_south_pole = smooth_south_pole_sum / CHARTS;

    for (int c = 0; c < CHARTS; c++) {
        for (int h = 0; h < H_PADDED; h++) {
            for (int w = 0; w < W_PADDED; w++) {
#pragma HLS PIPELINE II=1
                const int reorder_val = reorder_idx[0][c][h][w];
                const int src_chart = reorder_val / (H * W);
                const int remainder = reorder_val % (H * W);
                const int src_h = remainder / W;
                const int src_w = remainder % W;
                output[0][c][h][w] = smoothed_input[src_chart][src_h][src_w];
            }
        }
    }

    for (int c = 0; c < CHARTS; c++) {
        output[0][c][H_PADDED - 1][1] = smooth_north_pole;
        output[0][c][1][W_PADDED - 1] = smooth_south_pole;
    }
}

void get_kernel(
    const data_t weight[COUT][CIN][RIN][7],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    data_t kernel[COUT][ROUT][CIN][RIN][KERNEL_H][KERNEL_W]
) {
    for (int co = 0; co < COUT; co++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int ci = 0; ci < CIN; ci++) {
                for (int ri = 0; ri < RIN; ri++) {
                    for (int kh = 0; kh < KERNEL_H; kh++) {
#pragma HLS UNROLL
                        for (int kw = 0; kw < KERNEL_W; kw++) {
#pragma HLS UNROLL
                            kernel[co][ro][ci][ri][kh][kw] = 0.0f;
                        }
                    }
                }
            }
        }
    }

    for (int co = 0; co < COUT; co++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int ci = 0; ci < CIN; ci++) {
                for (int ri = 0; ri < RIN; ri++) {
                    for (int k = 0; k < 9; k++) {
                        const int idx_cout = kernel_expansion_idx[co][ro][ci][ri][k][0];
                        const int idx_cin = kernel_expansion_idx[co][ro][ci][ri][k][1];
                        const int idx_rin = kernel_expansion_idx[co][ro][ci][ri][k][2];
                        const int idx_w = kernel_expansion_idx[co][ro][ci][ri][k][3];
                        const int kh = k / 3;
                        const int kw = k % 3;

                        if (idx_w >= 0 && idx_w < 7) {
                            kernel[co][ro][ci][ri][kh][kw] = weight[idx_cout][idx_cin][idx_rin][idx_w];
                        }
                    }
                }
            }
        }
    }

    for (int co = 0; co < COUT; co++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int ci = 0; ci < CIN; ci++) {
                for (int ri = 0; ri < RIN; ri++) {
                    kernel[co][ro][ci][ri][0][2] = 0.0f;
                    kernel[co][ro][ci][ri][2][0] = 0.0f;
                }
            }
        }
    }
}

void conv2d_3x3(
    data_t input[(CIN * RIN)][(CHARTS * H_PADDED)][W_PADDED],
    const data_t kernel[(COUT * ROUT)][(CIN * RIN)][KERNEL_H][KERNEL_W],
    const data_t bias[COUT * ROUT],
    data_t output[(COUT * ROUT)][(CHARTS * H_PADDED)][W_PADDED]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=kernel cyclic factor=OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OC_PAR_FACTOR dim=1

    const int in_ch = CIN * RIN;
    const int out_ch = COUT * ROUT;
    const int in_h = CHARTS * H_PADDED;
    const int in_w = W_PADDED;

    for (int oc = 0; oc < out_ch; oc++) {
#pragma HLS UNROLL factor=OC_PAR_FACTOR
        for (int oh = 0; oh < in_h; oh++) {
            for (int ow = 0; ow < in_w; ow++) {
#pragma HLS PIPELINE II=1
                data_t sum = bias[oc];

                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kh = 0; kh < KERNEL_H; kh++) {
                        for (int kw = 0; kw < KERNEL_W; kw++) {
                            const int ih = oh + kh - 1;
                            const int iw = ow + kw - 1;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += input[ic][ih][iw] * kernel[oc][ic][kh][kw];
                            }
                        }
                    }
                }

                output[oc][oh][ow] = sum;
            }
        }
    }
}

static void post_process_and_writeback_output_frame(
    const data_t local_output[COUT][ROUT][CHARTS][H][W],
    data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W],
    int t
) {
    for (int co = 0; co < COUT; co++) {
        float north_mean[CHARTS];
        float south_mean[CHARTS];

        for (int c = 0; c < CHARTS; c++) {
            const int prev_c = (c - 1 + CHARTS) % CHARTS;
            float sum_v1 = 0.0f;
            float sum_v2 = 0.0f;
            for (int ro = 0; ro < ROUT; ro++) {
                sum_v1 += local_output[co][ro][c][1][0];
                sum_v1 += local_output[co][ro][c][1][1];
                sum_v1 += local_output[co][ro][c][0][1];
                sum_v1 += local_output[co][ro][prev_c][H - 1][H];
                sum_v1 += local_output[co][ro][prev_c][H - 1][H - 1];
                sum_v2 += local_output[co][ro][c][1][H];
                sum_v2 += local_output[co][ro][c][1][(H + 1) % W];
                sum_v2 += local_output[co][ro][c][0][(H + 1) % W];
                sum_v2 += local_output[co][ro][prev_c][H - 1][W - 1];
                sum_v2 += local_output[co][ro][c][0][H - 1];
            }
            north_mean[c] = sum_v1 / (ROUT * 5.0f);
            south_mean[c] = sum_v2 / (ROUT * 5.0f);
        }

        for (int ro = 0; ro < ROUT; ro++) {
            for (int c = 0; c < CHARTS; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
#pragma HLS PIPELINE II=1
                        data_t val = local_output[co][ro][c][h][w];
                        if (h == 0 && w == 0) {
                            val = north_mean[c];
                        } else if (h == 0 && w == H) {
                            val = south_mean[c];
                        }
                        output[t][co][ro][c][h][w] = val;
                    }
                }
            }
        }
    }
}

void conv_ico_layer0(
    data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W],
    const data_t weight[COUT][CIN][RIN][7],
    const data_t bias[COUT],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=OC_PAR_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=OC_PAR_FACTOR dim=1

    static data_t kernel[COUT][ROUT][CIN][RIN][KERNEL_H][KERNEL_W];
#pragma HLS ARRAY_PARTITION variable=kernel cyclic factor=OC_PAR_FACTOR dim=1
    get_kernel(weight, kernel_expansion_idx, kernel);

    static_assert(CIN == 1, "conv_ico_layer0 currently assumes CIN == 1.");
    static_assert(RIN == 1, "conv_ico_layer0 currently assumes RIN == 1.");

    for (int t = 0; t < TIME_STEPS; t++) {
        data_t padded_frame[RIN][CHARTS][H_PADDED][W_PADDED];
        static data_t local_output[COUT][ROUT][CHARTS][H][W];

        pad_ico(input[t], reorder_idx, padded_frame);

        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                for (int c = 0; c < CHARTS; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            data_t sum = bias[co];
                            for (int kh = 0; kh < KERNEL_H; kh++) {
                                for (int kw = 0; kw < KERNEL_W; kw++) {
                                    sum += padded_frame[0][c][h + kh][w + kw] * kernel[co][ro][0][0][kh][kw];
                                }
                            }
                            local_output[co][ro][c][h][w] = sum;
                        }
                    }
                }
            }
        }

        post_process_and_writeback_output_frame(local_output, output, t);
    }
}
