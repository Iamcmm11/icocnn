#include "ifan_temporal_r1.hpp"

#ifndef IFAN_TEMPORAL_CI_PAR_FACTOR
#define IFAN_TEMPORAL_CI_PAR_FACTOR 2
#endif

static inline weight_t temporal_to_weight_t(data_t x) {
    return static_cast<weight_t>(x);
}

static inline data_t temporal_to_data_t(acc_t x) {
    return static_cast<data_t>(x);
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
                staged_weight[coo][ci][k] =
                    (co < IFAN_BRANCH_CHANNELS) ? temporal_to_weight_t(weight[co][ci][k]) : static_cast<weight_t>(0);
            }
        }
    }
}

void ifan_temporal_r1_engine(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL],
    const data_t bias[IFAN_BRANCH_CHANNELS],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int ri = 0; ri < IFAN_R_FULL; ri++) {
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1; h++) {
                for (int w = 0; w < IFAN_W_R1; w++) {
                    input_t staged_input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=staged_input complete dim=2
                    input_t causal_window[IFAN_STAGE1_T][IFAN_TEMPORAL_KERNEL][IFAN_BRANCH_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=causal_window complete dim=2
#pragma HLS ARRAY_PARTITION variable=causal_window cyclic factor=IFAN_TEMPORAL_CI_PAR_FACTOR dim=3

                    for (int t = 0; t < IFAN_STAGE1_T; t++) {
                        for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
#pragma HLS PIPELINE II=1
                            staged_input[t][ci] = static_cast<input_t>(input[t][ci][ri][ch][h][w]);
                        }
                    }

                    for (int t = 0; t < IFAN_STAGE1_T; t++) {
                        for (int k = 0; k < IFAN_TEMPORAL_KERNEL; k++) {
                            const int src_t = t - (IFAN_TEMPORAL_KERNEL - 1) + k;
                            for (int ci = 0; ci < IFAN_BRANCH_CHANNELS; ci++) {
#pragma HLS PIPELINE II=1
                                causal_window[t][k][ci] =
                                    (src_t >= 0) ? staged_input[src_t][ci] : static_cast<input_t>(0);
                            }
                        }
                    }

                    for (int co_base = 0; co_base < IFAN_BRANCH_CHANNELS; co_base += IFAN_OC_TILE) {
                        weight_t staged_weight[IFAN_OC_TILE][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL];
#pragma HLS ARRAY_PARTITION variable=staged_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=staged_weight complete dim=3
#pragma HLS ARRAY_PARTITION variable=staged_weight cyclic factor=IFAN_TEMPORAL_CI_PAR_FACTOR dim=2
                        stage_temporal_weight_tile(weight, co_base, staged_weight);

                        for (int t = 0; t < IFAN_STAGE1_T; t++) {
#pragma HLS PIPELINE II=1
                            for (int coo = 0; coo < IFAN_OC_TILE; coo++) {
#pragma HLS UNROLL
                                const int co = co_base + coo;
                                if (co >= IFAN_BRANCH_CHANNELS) {
                                    continue;
                                }
                                acc_t sum = static_cast<acc_t>(bias[co]);
                                for (int ci_base = 0; ci_base < IFAN_BRANCH_CHANNELS; ci_base += IFAN_TEMPORAL_CI_PAR_FACTOR) {
                                    for (int k = 0; k < IFAN_TEMPORAL_KERNEL; k++) {
#pragma HLS UNROLL
                                        for (int cio = 0; cio < IFAN_TEMPORAL_CI_PAR_FACTOR; cio++) {
#pragma HLS UNROLL
                                            const int ci = ci_base + cio;
                                            if (ci < IFAN_BRANCH_CHANNELS) {
                                                sum += static_cast<acc_t>(causal_window[t][k][ci]) *
                                                       static_cast<acc_t>(staged_weight[coo][ci][k]);
                                            }
                                        }
                                    }
                                }
                                output[t][co][ri][ch][h][w] = temporal_to_data_t(sum);
                            }
                        }
                    }
                }
            }
        }
    }
}

void ifan_temporal_r1_top(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL],
    const data_t bias[IFAN_BRANCH_CHANNELS],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=IFAN_OC_TILE dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=IFAN_OC_TILE dim=1
    ifan_temporal_r1_engine(input, weight, bias, output);
}
