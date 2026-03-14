#include "ico_conv_layer1.hpp"

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
                int prev_c = (c - 1 + CHARTS) % CHARTS;

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

void pad_ico(
    data_t input[CIN][RIN][CHARTS][H][W],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[CIN][RIN][CHARTS][H_PADDED][W_PADDED]
) {
    static data_t input_after_smooth[CIN][RIN][CHARTS][H][W];
    smooth_vertices(input, input_after_smooth);

    for (int ci = 0; ci < CIN; ci++) {
        float smooth_north_pole_sum = 0.0f;
        float smooth_south_pole_sum = 0.0f;

        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                smooth_north_pole_sum += input_after_smooth[ci][ri][c][H - 1][0];
                smooth_south_pole_sum += input_after_smooth[ci][ri][c][0][W - 1];
            }
        }
        float smooth_north_pole = smooth_north_pole_sum / (RIN * CHARTS);
        float smooth_south_pole = smooth_south_pole_sum / (RIN * CHARTS);

        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                for (int h = 0; h < H_PADDED; h++) {
                    for (int w = 0; w < W_PADDED; w++) {
                        int reorder_val = reorder_idx[ri][c][h][w];
                        int src_chart = reorder_val / (H * W);
                        int remainder = reorder_val % (H * W);
                        int src_h = remainder / W;
                        int src_w = remainder % W;
                        output[ci][ri][c][h][w] = input_after_smooth[ci][ri][src_chart][src_h][src_w];
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
                        int idx_cout = kernel_expansion_idx[co][ro][ci][ri][k][0];
                        int idx_cin = kernel_expansion_idx[co][ro][ci][ri][k][1];
                        int idx_rin = kernel_expansion_idx[co][ro][ci][ri][k][2];
                        int idx_w = kernel_expansion_idx[co][ro][ci][ri][k][3];

                        int kh = k / 3;
                        int kw = k % 3;

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

void conv_ico_layer1(
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

    for (int t = 0; t < TIME_STEPS; t++) {
        static data_t padded_frame[CIN][RIN][CHARTS][H_PADDED][W_PADDED];
        pad_ico(input[t], reorder_idx, padded_frame);

        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                for (int c = 0; c < CHARTS; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            data_t sum = bias[co];
                            for (int ci = 0; ci < CIN; ci++) {
                                for (int ri = 0; ri < RIN; ri++) {
                                    for (int kh = 0; kh < KERNEL_H; kh++) {
                                        for (int kw = 0; kw < KERNEL_W; kw++) {
                                            int ph = h + kh;
                                            int pw = w + kw;
                                            sum += padded_frame[ci][ri][c][ph][pw] * kernel[co][ro][ci][ri][kh][kw];
                                        }
                                    }
                                }
                            }
                            output[t][co][ro][c][h][w] = sum;
                        }
                    }
                }
            }
        }

        for (int co = 0; co < COUT; co++) {
            for (int ro = 0; ro < ROUT; ro++) {
                for (int c = 0; c < CHARTS; c++) {
                    output[t][co][ro][c][0][0] = 0.0f;
                    output[t][co][ro][c][0][H] = 0.0f;
                }
            }
        }

        for (int co = 0; co < COUT; co++) {
            for (int c = 0; c < CHARTS; c++) {
                int prev_c = (c - 1 + CHARTS) % CHARTS;
                float sum_v1 = 0.0f;
                float sum_v2 = 0.0f;
                for (int ro = 0; ro < ROUT; ro++) {
                    sum_v1 += output[t][co][ro][c][1][0];
                    sum_v1 += output[t][co][ro][c][1][1];
                    sum_v1 += output[t][co][ro][c][0][1];
                    sum_v1 += output[t][co][ro][prev_c][H - 1][H];
                    sum_v1 += output[t][co][ro][prev_c][H - 1][H - 1];
                    sum_v2 += output[t][co][ro][c][1][H];
                    sum_v2 += output[t][co][ro][c][1][(H + 1) % W];
                    sum_v2 += output[t][co][ro][c][0][(H + 1) % W];
                    sum_v2 += output[t][co][ro][prev_c][H - 1][W - 1];
                    sum_v2 += output[t][co][ro][c][0][H - 1];
                }
                float mean_v1 = sum_v1 / (ROUT * 5.0f);
                float mean_v2 = sum_v2 / (ROUT * 5.0f);
                for (int ro = 0; ro < ROUT; ro++) {
                    output[t][co][ro][c][0][0] = mean_v1;
                    output[t][co][ro][c][0][H] = mean_v2;
                }
            }
        }
    }
}
