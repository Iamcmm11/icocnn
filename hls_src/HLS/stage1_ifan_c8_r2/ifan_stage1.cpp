#include "ifan_stage1.hpp"
#include "ifan_stage1_engines.hpp"

static void extract_feature_channel(
    data_t input[IFAN_IN_CHANNELS][IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    int feature_channel,
    data_t output[IFAN_STAGE1_T][1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
            for (int h = 0; h < IFAN_H_R2; h++) {
                for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                    output[t][0][0][ch][h][w] = input[feature_channel][t][ch][h][w];
                }
            }
        }
    }
}

static void residual_add_relu_r2(
    data_t residual[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    data_t x[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    for (int t = 0; t < IFAN_STAGE1_T; t++) {
        for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
            for (int r = 0; r < IFAN_R_FULL; r++) {
                for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                    for (int h = 0; h < IFAN_H_R2; h++) {
                        for (int w = 0; w < IFAN_W_R2; w++) {
#pragma HLS PIPELINE II=1
                            const data_t v = residual[t][c][r][ch][h][w] + x[t][c][r][ch][h][w];
                            output[t][c][r][ch][h][w] = v > 0.0f ? v : 0.0f;
                        }
                    }
                }
            }
        }
    }
}

static void frontend_branch_engine(
    data_t branch_input[IFAN_STAGE1_T][1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const data_t stem_w[IFAN_BRANCH_CHANNELS][1][1][IFAN_KERNEL_NEIGHBORS],
    const data_t stem_b[IFAN_BRANCH_CHANNELS],
    const data_t res_w[2][IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS],
    const data_t res_b[2][IFAN_BRANCH_CHANNELS],
    const data_t norm_gamma[IFAN_BRANCH_CHANNELS],
    const data_t norm_beta[IFAN_BRANCH_CHANNELS],
    const int kernel_idx_stem[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][1][1][9][4],
    const int kernel_idx_main[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4],
    const int reorder_r2_stem[1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    const int reorder_r2_main[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    data_t direct[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    data_t enhanced[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    static data_t tmp0[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t tmp1[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];

    ico_conv_r2_stem_engine(branch_input, stem_w, stem_b, kernel_idx_stem, reorder_r2_stem, direct);
    relu_feature_r2(direct);

    ico_conv_r2_main_engine(direct, res_w[0], res_b[0], kernel_idx_main, reorder_r2_main, tmp0);
    relu_feature_r2(tmp0);
    ico_conv_r2_main_engine(tmp0, res_w[1], res_b[1], kernel_idx_main, reorder_r2_main, tmp1);
    lnorm_ico_r2_engine(tmp1, norm_gamma, norm_beta, tmp0);
    residual_add_relu_r2(direct, tmp0, enhanced);
}

static void shared_attention_engine(
    data_t enhanced[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const data_t attn_w[2][IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS],
    const data_t attn_b[2][IFAN_BRANCH_CHANNELS],
    const data_t norm_gamma[IFAN_BRANCH_CHANNELS],
    const data_t norm_beta[IFAN_BRANCH_CHANNELS],
    const int kernel_idx_main[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4],
    const int reorder_r2_main[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    data_t weight[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
) {
    static data_t tmp0[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t tmp1[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];

    lnorm_ico_r2_engine(enhanced, norm_gamma, norm_beta, tmp0);
    ico_conv_r2_main_engine(tmp0, attn_w[0], attn_b[0], kernel_idx_main, reorder_r2_main, tmp1);
    relu_feature_r2(tmp1);
    ico_conv_r2_main_engine(tmp1, attn_w[1], attn_b[1], kernel_idx_main, reorder_r2_main, weight);
    sigmoid_feature_r2(weight);
}

static void fusion_block_engine(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t conv_w[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS],
    const data_t conv_b[IFAN_BRANCH_CHANNELS],
    const data_t temporal_w[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL],
    const data_t temporal_b[IFAN_BRANCH_CHANNELS],
    const data_t norm_gamma[IFAN_BRANCH_CHANNELS],
    const data_t norm_beta[IFAN_BRANCH_CHANNELS],
    const int kernel_idx_main[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4],
    const int reorder_r1[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1 + 2][IFAN_W_R1 + 2],
    bool apply_relu,
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
    static data_t tmp0[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
    static data_t tmp1[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];

    ico_conv_r1_main_engine(input, conv_w, conv_b, kernel_idx_main, reorder_r1, tmp0);
    relu_feature_r1(tmp0);
    temporal_conv1d_r1_engine(tmp0, temporal_w, temporal_b, tmp1);
    lnorm_ico_r1_engine(tmp1, norm_gamma, norm_beta, output);
    if (apply_relu) {
        relu_feature_r1(output);
    }
}

void ifan_stage1_top(
    data_t input[IFAN_IN_CHANNELS][IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const IfanStage1Weights &weights,
    const int reorder_r2_stem[1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    const int reorder_r2_main[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    const int reorder_r1[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1 + 2][IFAN_W_R1 + 2],
    const int kernel_idx_stem[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][1][1][9][4],
    const int kernel_idx_main[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
) {
#pragma HLS INLINE off

    static data_t phat_input[IFAN_STAGE1_T][1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t lms_input[IFAN_STAGE1_T][1][1][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];

    static data_t phat_direct[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t phat_enhanced[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t lms_direct[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t lms_enhanced[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t phat_attention[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t lms_attention[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t phat_fused[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t lms_fused[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t fused_r2[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2];
    static data_t fused_r1_a[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
    static data_t fused_r1_b[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];

    extract_feature_channel(input, 0, phat_input);
    extract_feature_channel(input, 1, lms_input);

    frontend_branch_engine(
        phat_input,
        weights.phat_stem_w,
        weights.phat_stem_b,
        weights.phat_res_w,
        weights.phat_res_b,
        weights.norm_gamma[0],
        weights.norm_beta[0],
        kernel_idx_stem,
        kernel_idx_main,
        reorder_r2_stem,
        reorder_r2_main,
        phat_direct,
        phat_enhanced
    );

    frontend_branch_engine(
        lms_input,
        weights.lms_stem_w,
        weights.lms_stem_b,
        weights.lms_res_w,
        weights.lms_res_b,
        weights.norm_gamma[1],
        weights.norm_beta[1],
        kernel_idx_stem,
        kernel_idx_main,
        reorder_r2_stem,
        reorder_r2_main,
        lms_direct,
        lms_enhanced
    );

    shared_attention_engine(
        phat_enhanced,
        weights.attn_w,
        weights.attn_b,
        weights.norm_gamma[2],
        weights.norm_beta[2],
        kernel_idx_main,
        reorder_r2_main,
        phat_attention
    );
    shared_attention_engine(
        lms_enhanced,
        weights.attn_w,
        weights.attn_b,
        weights.norm_gamma[2],
        weights.norm_beta[2],
        kernel_idx_main,
        reorder_r2_main,
        lms_attention
    );

    attention_fuse_r2(phat_direct, phat_enhanced, phat_attention, phat_fused);
    attention_fuse_r2(lms_direct, lms_enhanced, lms_attention, lms_fused);
    add_feature_r2(phat_fused, lms_fused, fused_r2);

    pool_ico_r2_to_r1_engine(
        fused_r2,
        reorder_r2_main,
        fused_r1_a
    );

    for (int block = 0; block < 4; block++) {
        fusion_block_engine(
            fused_r1_a,
            weights.fusion_w[block],
            weights.fusion_b[block],
            weights.fusion_temporal_w[block],
            weights.fusion_temporal_b[block],
            weights.norm_gamma[3 + block],
            weights.norm_beta[3 + block],
            kernel_idx_main,
            reorder_r1,
            true,
            fused_r1_b
        );
        if (block < 3) {
            for (int t = 0; t < IFAN_STAGE1_T; t++) {
                for (int c = 0; c < IFAN_BRANCH_CHANNELS; c++) {
                    for (int r = 0; r < IFAN_R_FULL; r++) {
                        for (int ch = 0; ch < IFAN_CHARTS; ch++) {
                            for (int h = 0; h < IFAN_H_R1; h++) {
                                for (int w = 0; w < IFAN_W_R1; w++) {
#pragma HLS PIPELINE II=1
                                    fused_r1_a[t][c][r][ch][h][w] = fused_r1_b[t][c][r][ch][h][w];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fusion_block_engine(
        fused_r1_b,
        weights.final_w,
        weights.final_b,
        weights.final_temporal_w,
        weights.final_temporal_b,
        weights.norm_gamma[7],
        weights.norm_beta[7],
        kernel_idx_main,
        reorder_r1,
        false,
        output
    );
}
