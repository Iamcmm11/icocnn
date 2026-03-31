#include "ico_conv_layer2_5.hpp"

static inline input_t to_input_t(data_t x) {
    return static_cast<input_t>(x);
}

static inline weight_t to_weight_t(data_t x) {
    return static_cast<weight_t>(x);
}

static inline act_t to_act_t(data_t x) {
    return static_cast<act_t>(x);
}

static inline acc_t to_acc_t(data_t x) {
    return static_cast<acc_t>(x);
}

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
            const int prev_c = (c - 1 + CHARTS) % CHARTS;
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
            const float mean_v1 = sum_v1 / (RIN * 5.0f);
            const float mean_v2 = sum_v2 / (RIN * 5.0f);
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
            const int prev_c = (c - 1 + CHARTS) % CHARTS;
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

        const float smooth_north_pole = smooth_north_pole_sum / (RIN * CHARTS);
        const float smooth_south_pole = smooth_south_pole_sum / (RIN * CHARTS);

        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                for (int h = 0; h < H_PADDED; h++) {
                    for (int w = 0; w < W_PADDED; w++) {
                        const int reorder_val = reorder_idx[ri][c][h][w];
                        const int src_ri = reorder_val / (CHARTS * H * W);
                        const int remainder_ri = reorder_val % (CHARTS * H * W);
                        const int src_chart = remainder_ri / (H * W);
                        const int remainder = remainder_ri % (H * W);
                        const int src_h = remainder / W;
                        const int src_w = remainder % W;

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

static void stage_input_frame_quantized(
    data_t input[CIN][RIN][CHARTS][H][W],
    input_t staged[CIN][RIN][CHARTS][H][W]
) {
    for (int ci = 0; ci < CIN; ci++) {
        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
#pragma HLS PIPELINE II=1
                        staged[ci][ri][c][h][w] = to_input_t(input[ci][ri][c][h][w]);
                    }
                }
            }
        }
    }
}

static void pad_ico_quantized(
    input_t input[CIN][RIN][CHARTS][H][W],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    input_t output[CIN][RIN][CHARTS][H_PADDED][W_PADDED]
) {
    input_t north_vertex[CIN][CHARTS];
    input_t south_vertex[CIN][CHARTS];
    const acc_t vertex_neighbor_count = static_cast<acc_t>(RIN * 5);
    const acc_t pole_neighbor_count = static_cast<acc_t>(RIN * CHARTS);

    for (int ci = 0; ci < CIN; ci++) {
        acc_t smooth_north_pole_sum = 0;
        acc_t smooth_south_pole_sum = 0;

        for (int c = 0; c < CHARTS; c++) {
            const int prev_c = (c - 1 + CHARTS) % CHARTS;
            acc_t sum_v1 = 0;
            acc_t sum_v2 = 0;
            for (int ri = 0; ri < RIN; ri++) {
                sum_v1 += static_cast<acc_t>(input[ci][ri][c][1][0]);
                sum_v1 += static_cast<acc_t>(input[ci][ri][c][1][1]);
                sum_v1 += static_cast<acc_t>(input[ci][ri][c][0][1]);
                sum_v1 += static_cast<acc_t>(input[ci][ri][prev_c][H - 1][H]);
                sum_v1 += static_cast<acc_t>(input[ci][ri][prev_c][H - 1][H - 1]);

                sum_v2 += static_cast<acc_t>(input[ci][ri][c][1][H]);
                sum_v2 += static_cast<acc_t>(input[ci][ri][c][1][(H + 1) % W]);
                sum_v2 += static_cast<acc_t>(input[ci][ri][c][0][(H + 1) % W]);
                sum_v2 += static_cast<acc_t>(input[ci][ri][prev_c][H - 1][W - 1]);
                sum_v2 += static_cast<acc_t>(input[ci][ri][c][0][H - 1]);
            }
            north_vertex[ci][c] = static_cast<input_t>(sum_v1 / vertex_neighbor_count);
            south_vertex[ci][c] = static_cast<input_t>(sum_v2 / vertex_neighbor_count);
        }

        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                smooth_north_pole_sum += static_cast<acc_t>(input[ci][ri][c][H - 1][0]);
                smooth_south_pole_sum += static_cast<acc_t>(input[ci][ri][c][0][W - 1]);
            }
        }

        const input_t smooth_north_pole = static_cast<input_t>(smooth_north_pole_sum / pole_neighbor_count);
        const input_t smooth_south_pole = static_cast<input_t>(smooth_south_pole_sum / pole_neighbor_count);

        for (int ri = 0; ri < RIN; ri++) {
            for (int c = 0; c < CHARTS; c++) {
                for (int h = 0; h < H_PADDED; h++) {
                    for (int w = 0; w < W_PADDED; w++) {
#pragma HLS PIPELINE II=1
                        const int reorder_val = reorder_idx[ri][c][h][w];
                        const int src_ri = reorder_val / (CHARTS * H * W);
                        const int remainder_ri = reorder_val % (CHARTS * H * W);
                        const int src_chart = remainder_ri / (H * W);
                        const int remainder = remainder_ri % (H * W);
                        const int src_h = remainder / W;
                        const int src_w = remainder % W;

                        input_t val = input[ci][src_ri][src_chart][src_h][src_w];
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

static inline void load_expanded_kernel(
    const data_t weight[COUT][CIN][RIN][7],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    int co,
    int ro,
    int ci,
    int ri,
    weight_t kernel[KERNEL_H][KERNEL_W]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=0
    for (int kh = 0; kh < KERNEL_H; kh++) {
#pragma HLS UNROLL
        for (int kw = 0; kw < KERNEL_W; kw++) {
#pragma HLS UNROLL
            kernel[kh][kw] = to_weight_t(expanded_weight_at(weight, kernel_expansion_idx, co, ro, ci, ri, kh, kw));
        }
    }
}

static void init_output_tiles(
    const data_t bias[COUT],
    int co_base,
    act_t output_tile[OC_TILE][ROUT][CHARTS][H][W]
) {
    for (int coo = 0; coo < OC_TILE; coo++) {
        const int co = co_base + coo;
        const act_t bias_val = (co < COUT) ? to_act_t(bias[co]) : static_cast<act_t>(0);
        for (int ro = 0; ro < ROUT; ro++) {
            for (int sp = 0; sp < CHARTS * H * W; sp++) {
#pragma HLS PIPELINE II=1
                const int c = sp / (H * W);
                const int rem = sp % (H * W);
                const int h = rem / W;
                const int w = rem % W;
                output_tile[coo][ro][c][h][w] = bias_val;
            }
        }
    }
}

static void post_process_output_tiles(
    act_t output_tile[OC_TILE][ROUT][CHARTS][H][W],
    act_t output_post[OC_TILE][ROUT][CHARTS][H][W]
) {
    const acc_t output_vertex_neighbor_count = static_cast<acc_t>(ROUT * 5);

    for (int coo = 0; coo < OC_TILE; coo++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int sp = 0; sp < CHARTS * H * W; sp++) {
#pragma HLS PIPELINE II=1
                const int c = sp / (H * W);
                const int rem = sp % (H * W);
                const int h = rem / W;
                const int w = rem % W;
                output_post[coo][ro][c][h][w] = output_tile[coo][ro][c][h][w];
            }
        }
    }

    for (int coo = 0; coo < OC_TILE; coo++) {
        for (int ro = 0; ro < ROUT; ro++) {
            for (int c = 0; c < CHARTS; c++) {
                output_post[coo][ro][c][0][0] = 0.0f;
                output_post[coo][ro][c][0][H] = 0.0f;
            }
        }
    }

    for (int coo = 0; coo < OC_TILE; coo++) {
        for (int c = 0; c < CHARTS; c++) {
            const int prev_c = (c - 1 + CHARTS) % CHARTS;
            acc_t sum_v1 = 0;
            acc_t sum_v2 = 0;
            for (int ro = 0; ro < ROUT; ro++) {
                sum_v1 += static_cast<acc_t>(output_tile[coo][ro][c][1][0]);
                sum_v1 += static_cast<acc_t>(output_tile[coo][ro][c][1][1]);
                sum_v1 += static_cast<acc_t>(output_tile[coo][ro][c][0][1]);
                sum_v1 += static_cast<acc_t>(output_tile[coo][ro][prev_c][H - 1][H]);
                sum_v1 += static_cast<acc_t>(output_tile[coo][ro][prev_c][H - 1][H - 1]);
                sum_v2 += static_cast<acc_t>(output_tile[coo][ro][c][1][H]);
                sum_v2 += static_cast<acc_t>(output_tile[coo][ro][c][1][(H + 1) % W]);
                sum_v2 += static_cast<acc_t>(output_tile[coo][ro][c][0][(H + 1) % W]);
                sum_v2 += static_cast<acc_t>(output_tile[coo][ro][prev_c][H - 1][W - 1]);
                sum_v2 += static_cast<acc_t>(output_tile[coo][ro][c][0][H - 1]);
            }
            const act_t mean_v1 = static_cast<act_t>(sum_v1 / output_vertex_neighbor_count);
            const act_t mean_v2 = static_cast<act_t>(sum_v2 / output_vertex_neighbor_count);
            for (int ro = 0; ro < ROUT; ro++) {
                output_post[coo][ro][c][0][0] = mean_v1;
                output_post[coo][ro][c][0][H] = mean_v2;
            }
        }
    }
}

static void writeback_output_tiles(
    const act_t output_post[OC_TILE][ROUT][CHARTS][H][W],
    int co_base,
    data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W],
    int t
) {
    for (int coo = 0; coo < OC_TILE; coo++) {
        const int co = co_base + coo;
        if (co >= COUT) {
            continue;
        }
        for (int ro = 0; ro < ROUT; ro++) {
            for (int sp = 0; sp < CHARTS * H * W; sp++) {
#pragma HLS PIPELINE II=1
                const int c = sp / (H * W);
                const int rem = sp % (H * W);
                const int h = rem / W;
                const int w = rem % W;
                output[t][co][ro][c][h][w] = static_cast<data_t>(output_post[coo][ro][c][h][w]);
            }
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
        static input_t staged_input[CIN][RIN][CHARTS][H][W];
        static input_t padded_frame[CIN][RIN][CHARTS][H_PADDED][W_PADDED];
#pragma HLS ARRAY_PARTITION variable=staged_input complete dim=4
#pragma HLS ARRAY_PARTITION variable=staged_input complete dim=5
#pragma HLS ARRAY_PARTITION variable=padded_frame complete dim=4
#pragma HLS ARRAY_PARTITION variable=padded_frame complete dim=5

        stage_input_frame_quantized(input[t], staged_input);
        pad_ico_quantized(staged_input, reorder_idx, padded_frame);

        for (int co_base = 0; co_base < COUT; co_base += OC_TILE) {
            act_t output_tile[OC_TILE][ROUT][CHARTS][H][W];
            act_t output_post[OC_TILE][ROUT][CHARTS][H][W];
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=4
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=5
#pragma HLS ARRAY_PARTITION variable=output_post complete dim=1
#pragma HLS ARRAY_PARTITION variable=output_post complete dim=2
#pragma HLS ARRAY_PARTITION variable=output_post complete dim=4
#pragma HLS ARRAY_PARTITION variable=output_post complete dim=5

            init_output_tiles(bias, co_base, output_tile);

            for (int ro = 0; ro < ROUT; ro++) {
                for (int ci = 0; ci < CIN; ci++) {
                    acc_t ri_partial[OC_TILE][RIN][CHARTS][H][W];
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=1
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=2
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=4
#pragma HLS ARRAY_PARTITION variable=ri_partial complete dim=5

                    for (int ri = 0; ri < RIN; ri++) {
                        weight_t kernel_tile[OC_TILE][KERNEL_H][KERNEL_W];
#pragma HLS ARRAY_PARTITION variable=kernel_tile complete dim=0
#pragma HLS ARRAY_PARTITION variable=kernel_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=kernel_tile complete dim=3
                        for (int coo = 0; coo < OC_TILE; coo++) {
#pragma HLS UNROLL
                            const int co = co_base + coo;
                            if (co < COUT) {
                                load_expanded_kernel(weight, kernel_expansion_idx, co, ro, ci, ri, kernel_tile[coo]);
                            } else {
                                for (int kh = 0; kh < KERNEL_H; kh++) {
#pragma HLS UNROLL
                                    for (int kw = 0; kw < KERNEL_W; kw++) {
#pragma HLS UNROLL
                                        kernel_tile[coo][kh][kw] = 0;
                                    }
                                }
                            }
                        }

                        for (int sp = 0; sp < CHARTS * H * W; sp++) {
#pragma HLS PIPELINE II=1
                            const int c = sp / (H * W);
                            const int rem = sp % (H * W);
                            const int h = rem / W;
                            const int w = rem % W;

                            for (int coo = 0; coo < OC_TILE; coo++) {
#pragma HLS UNROLL
                                const acc_t conv =
                                    static_cast<acc_t>(padded_frame[ci][ri][c][h + 0][w + 0]) * static_cast<acc_t>(kernel_tile[coo][0][0]) +
                                    static_cast<acc_t>(padded_frame[ci][ri][c][h + 0][w + 1]) * static_cast<acc_t>(kernel_tile[coo][0][1]) +
                                    static_cast<acc_t>(padded_frame[ci][ri][c][h + 0][w + 2]) * static_cast<acc_t>(kernel_tile[coo][0][2]) +
                                    static_cast<acc_t>(padded_frame[ci][ri][c][h + 1][w + 0]) * static_cast<acc_t>(kernel_tile[coo][1][0]) +
                                    static_cast<acc_t>(padded_frame[ci][ri][c][h + 1][w + 1]) * static_cast<acc_t>(kernel_tile[coo][1][1]) +
                                    static_cast<acc_t>(padded_frame[ci][ri][c][h + 1][w + 2]) * static_cast<acc_t>(kernel_tile[coo][1][2]) +
                                    static_cast<acc_t>(padded_frame[ci][ri][c][h + 2][w + 0]) * static_cast<acc_t>(kernel_tile[coo][2][0]) +
                                    static_cast<acc_t>(padded_frame[ci][ri][c][h + 2][w + 1]) * static_cast<acc_t>(kernel_tile[coo][2][1]) +
                                    static_cast<acc_t>(padded_frame[ci][ri][c][h + 2][w + 2]) * static_cast<acc_t>(kernel_tile[coo][2][2]);
                                ri_partial[coo][ri][c][h][w] = conv;
                            }
                        }
                    }

                    for (int sp = 0; sp < CHARTS * H * W; sp++) {
#pragma HLS PIPELINE II=1
                        const int c = sp / (H * W);
                        const int rem = sp % (H * W);
                        const int h = rem / W;
                        const int w = rem % W;

                        for (int coo = 0; coo < OC_TILE; coo++) {
#pragma HLS UNROLL
                            acc_t ci_sum = 0;
                            for (int ri = 0; ri < RIN; ri++) {
#pragma HLS UNROLL
                                ci_sum += ri_partial[coo][ri][c][h][w];
                            }
                            const acc_t accum = static_cast<acc_t>(output_tile[coo][ro][c][h][w]) + ci_sum;
                            output_tile[coo][ro][c][h][w] = static_cast<act_t>(accum);
                        }
                    }
                }
            }

            post_process_output_tiles(output_tile, output_post);
            writeback_output_tiles(output_post, co_base, output, t);
        }
    }
}
