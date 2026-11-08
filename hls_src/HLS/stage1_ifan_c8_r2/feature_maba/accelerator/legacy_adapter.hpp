#pragma once
#include "maba_top.hpp"
#include "../ifan_stage1_maba.hpp"
// Host-side packing only. Folded coefficients are computed in double then rounded once.
inline void maba_pack_legacy(const FeatureMabaWeights &w,float packed[maba_hw::WCOUNT]) {
 int i=0;
#define PACK2(F,A,B) for(int a=0;a<A;++a)for(int b=0;b<B;++b)packed[i++]=w.F[a][b];
#define PACK1(F,A) for(int a=0;a<A;++a)packed[i++]=w.F[a];
 PACK2(in_proj_weight,16,8) PACK1(in_proj_bias,16)
 for(int d=0;d<16;++d)for(int k=0;k<3;++k)packed[i++]=w.dw_conv_weight[d][0][k];
 PACK1(dw_conv_bias,16) PACK1(mix_norm_weight,16) PACK1(mix_norm_bias,16)
 PACK2(state_proj_weight,16,16) PACK1(state_proj_bias,16)
 PACK2(state_back_weight,16,8) PACK1(state_back_bias,16)
 PACK2(out_proj_weight,8,16) PACK1(out_proj_bias,8)
 for(int c=0;c<8;++c)for(int s=0;s<8;++s) {
  double a=0;for(int d=0;d<16;++d)a+=double(w.out_proj_weight[c][d])*w.state_back_weight[d][s];
  packed[i++]=float(a);
 }
 for(int c=0;c<8;++c) {
  double a=0;for(int d=0;d<16;++d)a+=double(w.out_proj_weight[c][d])*w.state_back_bias[d];
  packed[i++]=float(double(w.out_proj_bias[c])+a);
 }
#undef PACK1
#undef PACK2
}
inline void maba_legacy_adapter(
 float input[IFAN_STAGE1_T][8][6][5][2][4],const float packed[maba_hw::WCOUNT],
 float output[IFAN_STAGE1_T][8][6][5][2][4]) {
 static_assert(IFAN_STAGE1_T==MABA_T,"time dimension mismatch");
 static_assert(maba_hw::P==240,"legacy adapter requires the full spatial tensor");
 maba_top(&input[0][0][0][0][0][0],packed,&output[0][0][0][0][0][0]);
}
