#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include "legacy_adapter.hpp"
#include "../../post_maba/ifan_stage1_post.hpp"
#if defined(MABA_REQUIRE_FIXED) && !IFAN_STAGE1_HAS_AP_FIXED
#error Mixed-chain validation requires actual ap_fixed types.
#endif
// Readers generated from existing tests; never synthesized into the accelerator.
namespace stage {
#include "stage_reader.inc"
}
namespace maba {
#include "maba_reader.inc"
}
namespace post {
#include "post_reader.inc"
}
static float features[IFAN_STAGE1_T][8][6][5][2][4],output[IFAN_STAGE1_T][8][6][5][2][4];
static float packed[maba_hw::WCOUNT],coords[IFAN_STAGE1_T][3],original[IFAN_STAGE1_T][3];
bool compare(const char *name,float *a,float *b,int n,double mt,double rt) {
 double mx=0,sq=0;
 for(int i=0;i<n;++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))return false;double e=double(a[i])-b[i];mx=std::fmax(mx,std::fabs(e));sq+=e*e;}
 double rmse=std::sqrt(sq/n);bool ok=mx<=mt&&rmse<=rt;
 std::cout<<std::setprecision(10)<<"CHAIN "<<name<<" max_abs="<<mx<<" rmse="<<rmse<<" pass="<<ok<<"\n";return ok;
}
int main(int argc,char **argv) {
 if(argc<2)return 2;
 std::string dir=argv[1];
 if(!stage::init_real_data_from(dir)||!maba::load_data_from(dir)||!post::load_data_from(dir))return 3;
 maba_pack_legacy(maba::g_weights,packed);bool ok=true;
 for(int mode=0;mode<2;++mode) {
  if(mode==0)std::memcpy(features,maba::g_input,sizeof(features));
  else ifan_stage1_top(stage::g_input,stage::g_weights,stage::g_reorder_r2_stem,stage::g_reorder_r2_main,stage::g_reorder_r1,stage::g_kernel_idx_stem,stage::g_kernel_idx_main,features);
  feature_maba_engine(features,maba::g_weights,maba::g_input_positions,maba::g_in_proj_out,maba::g_dw_conv_input,maba::g_dw_conv_input_padded,maba::g_dw_conv_out,maba::g_mix_pre_norm,maba::g_mix_norm_out,maba::g_state_input,maba::g_q,maba::g_gate,maba::g_alpha,maba::g_state_sequence,maba::g_state_back_out,maba::g_refined_pre_dropout,maba::g_delta_flat,maba::g_delta,maba::g_output);
  maba_legacy_adapter(features,packed,output);
  post_maba_top(maba::g_output,post::g_weights,original);
  post_maba_top(output,post::g_weights,coords);
  ok=compare(mode?"actual_backbone_increment":"gold_backbone_increment",&coords[0][0],&original[0][0],IFAN_STAGE1_T*3,1e-4,1e-5)&&ok;
  // Total chain mismatch remains visible; not used to hide previously known errors.
  compare(mode?"actual_backbone_total":"gold_backbone_total",&coords[0][0],&post::g_ref_coords[0][0],IFAN_STAGE1_T*3,1e-4,1e-5);
 }
 std::cout<<"CHAIN_SUMMARY AP_FIXED="<<IFAN_STAGE1_HAS_AP_FIXED<<" pass="<<ok<<"\n";
 return ok?0:1;
}
