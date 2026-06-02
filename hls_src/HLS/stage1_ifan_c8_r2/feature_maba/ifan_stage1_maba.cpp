#include "ifan_stage1_maba.hpp"

static inline data_t maba_to_data(acc_t x) {
    return static_cast<data_t>(x);
}

static void flatten_feature_positions(
    data_t input[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t output[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS]
) {
    for (int r = 0; r < IFAN_R_FULL; r++) {
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1; h++) {
                for (int w = 0; w < IFAN_W_R1; w++) {
                    const int pos = (((r * IFAN_CHARTS + ch) * IFAN_H_R1 + h) * IFAN_W_R1 + w);
                    for (int t = 0; t < IFAN_STAGE1_T; t++) {
                        for (int c = 0; c < IFAN_MABA_CHANNELS; c++) {
#pragma HLS PIPELINE II=1
                            output[pos][t][c] = input[t][c][r][ch][h][w];
                        }
                    }
                }
            }
        }
    }
}

static void in_projection(
    data_t input_positions[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS],
    const data_t weight[IFAN_MABA_D_MODEL][IFAN_MABA_CHANNELS],
    const data_t bias[IFAN_MABA_D_MODEL],
    data_t output[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
#pragma HLS PIPELINE II=1
                acc_t sum = static_cast<acc_t>(bias[d]);
                for (int c = 0; c < IFAN_MABA_CHANNELS; c++) {
                    sum += static_cast<acc_t>(input_positions[pos][t][c]) *
                           static_cast<acc_t>(weight[d][c]);
                }
                output[pos][t][d] = maba_to_data(sum);
            }
        }
    }
}

static void transpose_for_depthwise(
    data_t input[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t output[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
            for (int t = 0; t < IFAN_STAGE1_T; t++) {
#pragma HLS PIPELINE II=1
                output[pos][d][t] = input[pos][t][d];
            }
        }
    }
}

static void causal_pad_depthwise_input(
    data_t input[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T],
    data_t output[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T + IFAN_MABA_CONV_KERNEL - 1]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
            for (int t = 0; t < IFAN_STAGE1_T + IFAN_MABA_CONV_KERNEL - 1; t++) {
#pragma HLS PIPELINE II=1
                if (t < IFAN_MABA_CONV_KERNEL - 1) {
                    output[pos][d][t] = 0.0f;
                } else {
                    output[pos][d][t] = input[pos][d][t - (IFAN_MABA_CONV_KERNEL - 1)];
                }
            }
        }
    }
}

static void depthwise_causal_conv(
    data_t input_padded[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T + IFAN_MABA_CONV_KERNEL - 1],
    const data_t weight[IFAN_MABA_D_MODEL][1][IFAN_MABA_CONV_KERNEL],
    const data_t bias[IFAN_MABA_D_MODEL],
    data_t output[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
#pragma HLS PIPELINE II=1
                acc_t sum = static_cast<acc_t>(bias[d]);
                for (int k = 0; k < IFAN_MABA_CONV_KERNEL; k++) {
                    sum += static_cast<acc_t>(input_padded[pos][d][t + k]) *
                           static_cast<acc_t>(weight[d][0][k]);
                }
                output[pos][t][d] = maba_to_data(sum);
            }
        }
    }
}

static void add_mix_pre_norm(
    data_t in_proj_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t dw_conv_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t output[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
#pragma HLS PIPELINE II=1
                output[pos][t][d] = in_proj_out[pos][t][d] + dw_conv_out[pos][t][d];
            }
        }
    }
}

static void layer_norm_d_model(
    data_t input[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    const data_t gamma[IFAN_MABA_D_MODEL],
    const data_t beta[IFAN_MABA_D_MODEL],
    data_t output[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            acc_t mean = 0;
            for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
                mean += static_cast<acc_t>(input[pos][t][d]);
            }
            mean /= static_cast<acc_t>(IFAN_MABA_D_MODEL);

            acc_t var = 0;
            for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
                const acc_t diff = static_cast<acc_t>(input[pos][t][d]) - mean;
                var += diff * diff;
            }
            var /= static_cast<acc_t>(IFAN_MABA_D_MODEL);
            const data_t inv_std = 1.0f / std::sqrt(static_cast<data_t>(var) + 1.0e-5f);

            for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
#pragma HLS PIPELINE II=1
                const data_t normed = (input[pos][t][d] - static_cast<data_t>(mean)) * inv_std;
                output[pos][t][d] = normed * gamma[d] + beta[d];
            }
        }
    }
}

static inline data_t sigmoid_scalar(data_t x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static void state_projection(
    data_t input[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    const data_t weight[IFAN_MABA_STATE_DIM * 2][IFAN_MABA_D_MODEL],
    const data_t bias[IFAN_MABA_STATE_DIM * 2],
    data_t output[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM * 2]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            for (int d = 0; d < IFAN_MABA_STATE_DIM * 2; d++) {
#pragma HLS PIPELINE II=1
                acc_t sum = static_cast<acc_t>(bias[d]);
                for (int c = 0; c < IFAN_MABA_D_MODEL; c++) {
                    sum += static_cast<acc_t>(input[pos][t][c]) *
                           static_cast<acc_t>(weight[d][c]);
                }
                output[pos][t][d] = maba_to_data(sum);
            }
        }
    }
}

static void split_gate_and_alpha(
    data_t state_input[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM * 2],
    data_t q[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM],
    data_t gate[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM],
    data_t alpha[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            for (int d = 0; d < IFAN_MABA_STATE_DIM; d++) {
#pragma HLS PIPELINE II=1
                q[pos][t][d] = state_input[pos][t][d];
                gate[pos][t][d] = state_input[pos][t][d + IFAN_MABA_STATE_DIM];
                alpha[pos][t][d] = sigmoid_scalar(gate[pos][t][d]);
            }
        }
    }
}

static void state_scan(
    data_t q[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM],
    data_t alpha[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM],
    data_t output[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        data_t state[IFAN_MABA_STATE_DIM];
        for (int d = 0; d < IFAN_MABA_STATE_DIM; d++) {
            state[d] = 0.0f;
        }
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            for (int d = 0; d < IFAN_MABA_STATE_DIM; d++) {
#pragma HLS PIPELINE II=1
                const data_t a = alpha[pos][t][d];
                state[d] = a * state[d] + (1.0f - a) * q[pos][t][d];
                output[pos][t][d] = state[d];
            }
        }
    }
}

static void state_back_projection(
    data_t input[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM],
    const data_t weight[IFAN_MABA_D_MODEL][IFAN_MABA_STATE_DIM],
    const data_t bias[IFAN_MABA_D_MODEL],
    data_t output[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
#pragma HLS PIPELINE II=1
                acc_t sum = static_cast<acc_t>(bias[d]);
                for (int s = 0; s < IFAN_MABA_STATE_DIM; s++) {
                    sum += static_cast<acc_t>(input[pos][t][s]) *
                           static_cast<acc_t>(weight[d][s]);
                }
                output[pos][t][d] = maba_to_data(sum);
            }
        }
    }
}

static void add_refined_pre_dropout(
    data_t mix_norm_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t state_back_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t output[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
#pragma HLS PIPELINE II=1
                output[pos][t][d] = mix_norm_out[pos][t][d] + state_back_out[pos][t][d];
            }
        }
    }
}

static void out_projection(
    data_t input[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    const data_t weight[IFAN_MABA_CHANNELS][IFAN_MABA_D_MODEL],
    const data_t bias[IFAN_MABA_CHANNELS],
    data_t output[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS]
) {
    for (int pos = 0; pos < IFAN_MABA_POSITIONS; pos++) {
        for (int t = 0; t < IFAN_STAGE1_T; t++) {
            for (int c = 0; c < IFAN_MABA_CHANNELS; c++) {
#pragma HLS PIPELINE II=1
                acc_t sum = static_cast<acc_t>(bias[c]);
                for (int d = 0; d < IFAN_MABA_D_MODEL; d++) {
                    sum += static_cast<acc_t>(input[pos][t][d]) *
                           static_cast<acc_t>(weight[c][d]);
                }
                output[pos][t][c] = maba_to_data(sum);
            }
        }
    }
}

static void unflatten_delta_and_residual(
    data_t input[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t delta_flat[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS],
    data_t delta[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t output[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int r = 0; r < IFAN_R_FULL; r++) {
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1; h++) {
                for (int w = 0; w < IFAN_W_R1; w++) {
                    const int pos = (((r * IFAN_CHARTS + ch) * IFAN_H_R1 + h) * IFAN_W_R1 + w);
                    for (int t = 0; t < IFAN_STAGE1_T; t++) {
                        for (int c = 0; c < IFAN_MABA_CHANNELS; c++) {
#pragma HLS PIPELINE II=1
                            const data_t v = delta_flat[pos][t][c];
                            delta[t][c][r][ch][h][w] = v;
                            output[t][c][r][ch][h][w] = input[t][c][r][ch][h][w] + v;
                        }
                    }
                }
            }
        }
    }
}

void feature_maba_engine(
    data_t input[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const FeatureMabaWeights &weights,
    data_t input_positions[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS],
    data_t in_proj_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t dw_conv_input[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T],
    data_t dw_conv_input_padded[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T + IFAN_MABA_CONV_KERNEL - 1],
    data_t dw_conv_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t mix_pre_norm[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t mix_norm_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t state_input[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM * 2],
    data_t q[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM],
    data_t gate[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM],
    data_t alpha[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM],
    data_t state_sequence[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM],
    data_t state_back_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t refined_pre_dropout[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL],
    data_t delta_flat[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS],
    data_t delta[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t output[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    flatten_feature_positions(input, input_positions);
    in_projection(input_positions, weights.in_proj_weight, weights.in_proj_bias, in_proj_out);
    transpose_for_depthwise(in_proj_out, dw_conv_input);
    causal_pad_depthwise_input(dw_conv_input, dw_conv_input_padded);
    depthwise_causal_conv(dw_conv_input_padded, weights.dw_conv_weight, weights.dw_conv_bias, dw_conv_out);
    add_mix_pre_norm(in_proj_out, dw_conv_out, mix_pre_norm);
    layer_norm_d_model(mix_pre_norm, weights.mix_norm_weight, weights.mix_norm_bias, mix_norm_out);
    state_projection(mix_norm_out, weights.state_proj_weight, weights.state_proj_bias, state_input);
    split_gate_and_alpha(state_input, q, gate, alpha);
    state_scan(q, alpha, state_sequence);
    state_back_projection(state_sequence, weights.state_back_weight, weights.state_back_bias, state_back_out);
    add_refined_pre_dropout(mix_norm_out, state_back_out, refined_pre_dropout);
    out_projection(refined_pre_dropout, weights.out_proj_weight, weights.out_proj_bias, delta_flat);
    unflatten_delta_and_residual(input, delta_flat, delta, output);
}
