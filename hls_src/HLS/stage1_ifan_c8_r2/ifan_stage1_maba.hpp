#ifndef IFAN_STAGE1_MABA_HPP
#define IFAN_STAGE1_MABA_HPP

#include "ifan_stage1.hpp"

#define IFAN_MABA_POSITIONS (IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1)
#define IFAN_MABA_CHANNELS IFAN_BRANCH_CHANNELS
#define IFAN_MABA_D_MODEL 16
#define IFAN_MABA_STATE_DIM 8
#define IFAN_MABA_CONV_KERNEL 3

struct FeatureMabaWeights {
    data_t in_proj_weight[IFAN_MABA_D_MODEL][IFAN_MABA_CHANNELS];
    data_t in_proj_bias[IFAN_MABA_D_MODEL];
    data_t dw_conv_weight[IFAN_MABA_D_MODEL][1][IFAN_MABA_CONV_KERNEL];
    data_t dw_conv_bias[IFAN_MABA_D_MODEL];
    data_t mix_norm_weight[IFAN_MABA_D_MODEL];
    data_t mix_norm_bias[IFAN_MABA_D_MODEL];
    data_t state_proj_weight[IFAN_MABA_STATE_DIM * 2][IFAN_MABA_D_MODEL];
    data_t state_proj_bias[IFAN_MABA_STATE_DIM * 2];
    data_t state_back_weight[IFAN_MABA_D_MODEL][IFAN_MABA_STATE_DIM];
    data_t state_back_bias[IFAN_MABA_D_MODEL];
    data_t out_proj_weight[IFAN_MABA_CHANNELS][IFAN_MABA_D_MODEL];
    data_t out_proj_bias[IFAN_MABA_CHANNELS];
};

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
);

#endif
