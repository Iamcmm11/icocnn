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
        for (int c = 0; c < CHARTS; c++) {
            int prev_c = (c - 1 + CHARTS) % CHARTS;

            // Match python icoCNN.SmoothVertices:
            // mean over both R dimension and 5 neighbors, then broadcast to all R.
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

    static_assert((CIN % IC_TILE) == 0, "CIN must be divisible by IC_TILE.");
    static_assert((COUT % OC_TILE) == 0, "COUT must be divisible by OC_TILE.");

    for (int t = 0; t < TIME_STEPS; t++) {
        static data_t padded_frame[CIN][RIN][CHARTS][H_PADDED][W_PADDED];
        pad_ico(input[t], reorder_idx, padded_frame);

        for (int co_base = 0; co_base < COUT; co_base += OC_TILE) {
            data_t psum[OC_TILE][ROUT][CHARTS][H][W];
#pragma HLS ARRAY_PARTITION variable=psum complete dim=1

            for (int co_t = 0; co_t < OC_TILE; co_t++) {
                for (int ro = 0; ro < ROUT; ro++) {
                    for (int c = 0; c < CHARTS; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                psum[co_t][ro][c][h][w] = bias[co_base + co_t];
                            }
                        }
                    }
                }
            }

            for (int ci_base = 0; ci_base < CIN; ci_base += IC_TILE) {
                for (int co_t = 0; co_t < OC_TILE; co_t++) {
                    int co = co_base + co_t;
                    for (int ro = 0; ro < ROUT; ro++) {
                        for (int c = 0; c < CHARTS; c++) {
                            for (int h = 0; h < H; h++) {
                                for (int w = 0; w < W; w++) {
                                    data_t sum = psum[co_t][ro][c][h][w];
                                    for (int ci_t = 0; ci_t < IC_TILE; ci_t++) {
                                        int ci = ci_base + ci_t;
                                        for (int ri = 0; ri < RIN; ri++) {
                                            for (int kh = 0; kh < KERNEL_H; kh++) {
                                                for (int kw = 0; kw < KERNEL_W; kw++) {
                                                    int ph = h + kh;
                                                    int pw = w + kw;
                                                    sum += padded_frame[ci][ri][c][ph][pw] *
                                                           expanded_weight_at(weight, kernel_expansion_idx, co, ro, ci, ri, kh, kw);
                                                }
                                            }
                                        }
                                    }
                                    psum[co_t][ro][c][h][w] = sum;
                                }
                            }
                        }
                    }
                }
            }

            for (int co_t = 0; co_t < OC_TILE; co_t++) {
                int co = co_base + co_t;
                for (int ro = 0; ro < ROUT; ro++) {
                    for (int c = 0; c < CHARTS; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                output[t][co][ro][c][h][w] = psum[co_t][ro][c][h][w];
                            }
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
