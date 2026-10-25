#include "ifan_v14_top.hpp"
void ifan_v14_top(
 data_t input[2][IFAN_STAGE1_T][5][4][8], const IfanStage1Weights &w,
 const int r0[1][5][6][10], const int r2[6][5][6][10], const int r1[6][5][4][6],
 const int k0[8][6][1][1][9][4], const int k1[8][6][8][6][9][4],
 const FeatureMabaWeights &mw, const PostMabaWeights &pw, data_t coords[IFAN_STAGE1_T][3]){
static data_t features[IFAN_STAGE1_T][8][6][5][2][4];
static data_t g_input_positions[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS];
static data_t g_in_proj_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_dw_conv_input[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T];
static data_t g_dw_conv_input_padded[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T + IFAN_MABA_CONV_KERNEL - 1];
static data_t g_dw_conv_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_mix_pre_norm[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_mix_norm_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_state_input[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM * 2];
static data_t g_q[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_gate[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_alpha[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_state_sequence[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_state_back_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_refined_pre_dropout[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_delta_flat[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS];
static data_t g_delta[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_output[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];


ifan_stage1_top(input,w,r0,r2,r1,k0,k1,features);
    feature_maba_engine(
        features,
        mw,
        g_input_positions,
        g_in_proj_out,
        g_dw_conv_input,
        g_dw_conv_input_padded,
        g_dw_conv_out,
        g_mix_pre_norm,
        g_mix_norm_out,
        g_state_input,
        g_q,
        g_gate,
        g_alpha,
        g_state_sequence,
        g_state_back_out,
        g_refined_pre_dropout,
        g_delta_flat,
        g_delta,
        g_output
    );

post_maba_top(g_output,pw,coords);
}
