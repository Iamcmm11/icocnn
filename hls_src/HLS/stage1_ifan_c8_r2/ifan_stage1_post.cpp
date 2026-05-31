#include "ifan_stage1_post.hpp"

static void channel_readout(
    data_t input[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t weight[IFAN_MABA_CHANNELS],
    data_t bias,
    data_t output[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int r = 0; r < IFAN_R_FULL; r++) {
            for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                for (int h = 0; h < IFAN_H_R1; h++) {
                    for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                        acc_t sum = static_cast<acc_t>(bias);
                        for (int c = 0; c < IFAN_MABA_CHANNELS; c++) {
                            sum += static_cast<acc_t>(input[t][c][r][ch][h][w]) *
                                   static_cast<acc_t>(weight[c]);
                        }
                        output[t][0][r][ch][h][w] = static_cast<data_t>(sum);
                    }
                }
            }
        }
    }
}

static void copy_post_final_pool(
    data_t input[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t output[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int r = 0; r < IFAN_R_FULL; r++) {
            for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                for (int h = 0; h < IFAN_H_R1; h++) {
                    for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                        output[t][0][r][ch][h][w] = input[t][0][r][ch][h][w];
                    }
                }
            }
        }
    }
}

static void region_max(
    data_t input[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t output[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    int argmax_idx[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1; h++) {
                for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                    data_t best = input[t][0][0][ch][h][w];
                    int best_r = 0;
                    for (int r = 1; r < IFAN_R_FULL; r++) {
                        const data_t v = input[t][0][r][ch][h][w];
                        if (v > best) {
                            best = v;
                            best_r = r;
                        }
                    }
                    output[t][ch][h][w] = best;
                    argmax_idx[t][ch][h][w] = best_r;
                }
            }
        }
    }
}

static void clean_vertices(
    data_t input[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t mask[IFAN_H_R1][IFAN_W_R1],
    data_t output[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1; h++) {
                for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                    output[t][ch][h][w] = input[t][ch][h][w] * mask[h][w];
                }
            }
        }
    }
}

static void softargmax(
    data_t input[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t indexes[IFAN_COORD_DIMS][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t prob[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t coords[IFAN_STAGE1_T][IFAN_COORD_DIMS]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        data_t max_v = input[t][0][0][0];
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1; h++) {
                for (int w = 0; w < IFAN_W_R1; w++) {
                    const data_t v = input[t][ch][h][w];
                    if (v > max_v) {
                        max_v = v;
                    }
                }
            }
        }

        acc_t denom = 0;
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1; h++) {
                for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                    const data_t e = std::exp(input[t][ch][h][w] - max_v);
                    prob[t][ch][h][w] = e;
                    denom += static_cast<acc_t>(e);
                }
            }
        }

        for (int dim = 0; dim < IFAN_COORD_DIMS; dim++) {
            coords[t][dim] = 0.0f;
        }

        const acc_t inv_denom = static_cast<acc_t>(1.0f) / (denom + static_cast<acc_t>(1.0e-12f));
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R1; h++) {
                for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                    const data_t p = static_cast<data_t>(static_cast<acc_t>(prob[t][ch][h][w]) * inv_denom);
                    prob[t][ch][h][w] = p;
                    for (int dim = 0; dim < IFAN_COORD_DIMS; dim++) {
                        coords[t][dim] += p * indexes[dim][ch][h][w];
                    }
                }
            }
        }
    }
}

void post_maba_engine(
    data_t input[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const PostMabaWeights &weights,
    data_t channel_readout_logits[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t post_final_pool_logits[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t region_max_logits[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    int region_argmax_idx[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t softargmax_input[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t softargmax_prob[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    data_t coords[IFAN_STAGE1_T][IFAN_COORD_DIMS]
) {
    channel_readout(input, weights.channel_readout_weight, weights.channel_readout_bias, channel_readout_logits);
    copy_post_final_pool(channel_readout_logits, post_final_pool_logits);
    region_max(post_final_pool_logits, region_max_logits, region_argmax_idx);
    clean_vertices(region_max_logits, weights.clean_vertices_mask, softargmax_input);
    softargmax(softargmax_input, weights.softargmax_indexes, softargmax_prob, coords);
}

void post_maba_top(
    data_t input[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const PostMabaWeights &weights,
    data_t coords[IFAN_STAGE1_T][IFAN_COORD_DIMS]
) {
    static data_t channel_readout_logits[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
    static data_t post_final_pool_logits[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
    static data_t region_max_logits[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
    static int region_argmax_idx[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
    static data_t softargmax_input[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
    static data_t softargmax_prob[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
    post_maba_engine(
        input,
        weights,
        channel_readout_logits,
        post_final_pool_logits,
        region_max_logits,
        region_argmax_idx,
        softargmax_input,
        softargmax_prob,
        coords
    );
}
