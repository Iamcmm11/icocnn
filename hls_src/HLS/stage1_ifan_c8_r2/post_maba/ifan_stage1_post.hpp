#ifndef IFAN_STAGE1_POST_HPP
#define IFAN_STAGE1_POST_HPP

#include "../feature_maba/ifan_stage1_maba.hpp"

#define IFAN_COORD_DIMS 3

struct PostMabaWeights {
    data_t channel_readout_weight[IFAN_MABA_CHANNELS];
    data_t channel_readout_bias;
    data_t clean_vertices_mask[IFAN_H_R1][IFAN_W_R1];
    data_t softargmax_indexes[IFAN_COORD_DIMS][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
};

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
);

void post_maba_top(
    data_t input[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const PostMabaWeights &weights,
    data_t coords[IFAN_STAGE1_T][IFAN_COORD_DIMS]
);

#endif
