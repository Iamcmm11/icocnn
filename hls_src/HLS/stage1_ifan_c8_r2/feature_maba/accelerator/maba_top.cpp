#include "maba_top.hpp"
#include <cmath>
#if MABA_VARIANT >= 2
#ifdef MABA_NATIVE
#include <queue>
namespace hls { template<class T> class stream {
 std::queue<T> q; public: explicit stream(const char*) {}
 void write(const T &v){q.push(v);} T read(){T v=q.front();q.pop();return v;}
}; }
#else
#include <hls_stream.h>
#endif
#endif
namespace maba_hw {
template<int NI, int NO>
void linear(const float x[NI],const float w[NO][NI],const float b[NO],float y[NO]) {
#pragma HLS INLINE
  for(int o=0;o<NO;++o) {
    float a=b[o];
    for(int i=0;i<NI;++i) {
#pragma HLS UNROLL factor=MABA_PAR
      a += x[i]*w[o][i];
    }
    y[o]=a;
  }
}
void norm(const float x[D],const Weights &w,float y[D]) {
#pragma HLS INLINE
  float mean=0, var=0;
  for(int d=0;d<D;++d) mean+=x[d];
  mean/=float(D);
  for(int d=0;d<D;++d) {float a=x[d]-mean;var+=a*a;}
  var/=float(D);
  float inv=1.0f/std::sqrt(var+1.0e-5f);
  for(int d=0;d<D;++d) y[d]=((x[d]-mean)*inv)*w.gamma[d]+w.beta[d];
}
void update(const float qg[2*S],float state[S],float h[S]) {
#pragma HLS INLINE
  for(int s=0;s<S;++s) {
    float a=1.0f/(1.0f+std::exp(-qg[S+s]));
    state[s]=a*state[s]+(1.0f-a)*qg[s];
    h[s]=state[s];
  }
}
void finish(const float x[C],const float z[D],const float h[S],const Weights &w,float y[C]) {
#pragma HLS INLINE
#if MABA_VARIANT == 4
  float a[C],b[C];
  const float zero[C]={0};
  linear<D,C>(z,w.wo,w.bf,a);
  linear<S,C>(h,w.wh,zero,b);
  for(int c=0;c<C;++c) y[c]=x[c]+(a[c]+b[c]);
#else
  float back[D],refined[D],delta[C];
  linear<S,D>(h,w.wb,w.bb,back);
  for(int d=0;d<D;++d) refined[d]=z[d]+back[d];
  linear<D,C>(refined,w.wo,w.bo,delta);
  for(int c=0;c<C;++c) y[c]=x[c]+delta[c];
#endif
}

// Stage-materialized baseline; exactly the same arithmetic as the streaming core.
template<int N>
void batch(const float input[COUNT],const Weights &w,float output[COUNT],int base,int n) {
  static float x[N][MABA_T][C], proj[N][MABA_T][D],mix[N][MABA_T][D];
  static float z[N][MABA_T][D],qg[N][MABA_T][2*S],hs[N][MABA_T][S];
  for(int t=0;t<MABA_T;++t) for(int c=0;c<C;++c) for(int p=0;p<n;++p) {
#pragma HLS PIPELINE II=1
    x[p][t][c]=input[(t*C+c)*P+base+p];
  }
  for(int p=0;p<n;++p) for(int t=0;t<MABA_T;++t)
    linear<C,D>(x[p][t],w.wi,w.bi,proj[p][t]);
  for(int p=0;p<n;++p) for(int t=0;t<MABA_T;++t) for(int d=0;d<D;++d) {
    float a=w.bc[d];
    for(int k=0;k<3;++k) a+=(t+k>=2?proj[p][t+k-2][d]:0.0f)*w.wc[d][k];
    mix[p][t][d]=proj[p][t][d]+a;
  }
  for(int p=0;p<n;++p) for(int t=0;t<MABA_T;++t) norm(mix[p][t],w,z[p][t]);
  for(int p=0;p<n;++p) for(int t=0;t<MABA_T;++t) linear<D,2*S>(z[p][t],w.ws,w.bs,qg[p][t]);
  for(int p=0;p<n;++p) {
    float state[S]={0};
    for(int t=0;t<MABA_T;++t) update(qg[p][t],state,hs[p][t]);
  }
  for(int p=0;p<n;++p) for(int t=0;t<MABA_T;++t) {
    float y[C];finish(x[p][t],z[p][t],hs[p][t],w,y);
    for(int c=0;c<C;++c) output[(t*C+c)*P+base+p]=y[c];
  }
}
#if MABA_VARIANT >= 2
struct InputPacket {float x[C];};
struct MixPacket {float x[C],z[D];};
struct StatePacket {float x[C],z[D],h[S];};
struct OutputPacket {float y[C];};
void position(int i,int n,int &p,int &t) {
#pragma HLS INLINE
#if MABA_VARIANT >= 3
  p=i%n;t=i/n;
#else
  p=i/MABA_T;t=i%MABA_T;
#endif
}
void load(const float input[COUNT],int base,int n,hls::stream<InputPacket> &out) {
  float tile[MABA_T][C][MABA_TILE];
  // Contiguous spatial bursts; permutation is paid inside the measured top.
  for(int t=0;t<MABA_T;++t) for(int c=0;c<C;++c) for(int p=0;p<n;++p) {
#pragma HLS PIPELINE II=1
    tile[t][c][p]=input[(t*C+c)*P+base+p];
  }
  for(int i=0;i<n*MABA_T;++i) {
    int p,t;position(i,n,p,t);InputPacket v;
    for(int c=0;c<C;++c) v.x[c]=tile[t][c][p];
    out.write(v);
  }
}
void project_conv(hls::stream<InputPacket> &in,hls::stream<MixPacket> &out,const Weights &w,int n) {
  float history[MABA_TILE][2][D];
  for(int i=0;i<n*MABA_T;++i) {
    int p,t;position(i,n,p,t);
    InputPacket a=in.read();MixPacket b;float projected[D];
    linear<C,D>(a.x,w.wi,w.bi,projected);
    for(int d=0;d<D;++d) {
      if(t==0) {history[p][0][d]=0;history[p][1][d]=0;}
      float v=w.bc[d];
      v+=history[p][0][d]*w.wc[d][0];
      v+=history[p][1][d]*w.wc[d][1];
      v+=projected[d]*w.wc[d][2];
      history[p][0][d]=history[p][1][d];history[p][1][d]=projected[d];
      b.z[d]=projected[d]+v;
    }
    for(int c=0;c<C;++c)b.x[c]=a.x[c];
    out.write(b);
  }
}
void normalize(hls::stream<MixPacket> &in,hls::stream<MixPacket> &out,const Weights &w,int n) {
  for(int i=0;i<n*MABA_T;++i) {
    MixPacket a=in.read(),b;norm(a.z,w,b.z);
    for(int c=0;c<C;++c)b.x[c]=a.x[c];
    out.write(b);
  }
}
void scan(hls::stream<MixPacket> &in,hls::stream<StatePacket> &out,const Weights &w,int n) {
  float state[MABA_TILE][S];
  for(int i=0;i<n*MABA_T;++i) {
    int p,t;position(i,n,p,t);MixPacket a=in.read();StatePacket b;float qg[2*S];
    linear<D,2*S>(a.z,w.ws,w.bs,qg);
    if(t==0)for(int s=0;s<S;++s)state[p][s]=0;
    update(qg,state[p],b.h);
    for(int c=0;c<C;++c)b.x[c]=a.x[c];
    for(int d=0;d<D;++d)b.z[d]=a.z[d];
    out.write(b);
  }
}
void readout(hls::stream<StatePacket> &in,hls::stream<OutputPacket> &out,const Weights &w,int n) {
  for(int i=0;i<n*MABA_T;++i) {
    StatePacket a=in.read();OutputPacket b;finish(a.x,a.z,a.h,w,b.y);out.write(b);
  }
}
void store(hls::stream<OutputPacket> &in,float output[COUNT],int base,int n) {
  float tile[MABA_T][C][MABA_TILE];
  for(int i=0;i<n*MABA_T;++i) {
    int p,t;position(i,n,p,t);OutputPacket a=in.read();
    for(int c=0;c<C;++c)tile[t][c][p]=a.y[c];
  }
  for(int t=0;t<MABA_T;++t)for(int c=0;c<C;++c)for(int p=0;p<n;++p) {
#pragma HLS PIPELINE II=1
    output[(t*C+c)*P+base+p]=tile[t][c][p];
  }
}
void streaming(const float input[COUNT],const Weights &w,float output[COUNT],int base,int n) {
#pragma HLS DATAFLOW
  hls::stream<InputPacket> a("input");
  hls::stream<MixPacket> b("mixed"),c("normalized");
  hls::stream<StatePacket> d("state");
  hls::stream<OutputPacket> e("output");
#pragma HLS STREAM variable=a depth=MABA_FIFO
#pragma HLS STREAM variable=b depth=MABA_FIFO
#pragma HLS STREAM variable=c depth=MABA_FIFO
#pragma HLS STREAM variable=d depth=MABA_FIFO
#pragma HLS STREAM variable=e depth=MABA_FIFO
  load(input,base,n,a);project_conv(a,b,w,n);normalize(b,c,w,n);
  scan(c,d,w,n);readout(d,e,w,n);store(e,output,base,n);
}
#endif
}
extern "C" void maba_top(const float input[maba_hw::COUNT],const float weights[maba_hw::WCOUNT],float output[maba_hw::COUNT]) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=maba_hw::COUNT
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem1 depth=maba_hw::WCOUNT
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2 depth=maba_hw::COUNT
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
  using namespace maba_hw;
  Weights w; int offset=0;
#define LOAD2(field,A,B) for(int a=0;a<A;++a)for(int b=0;b<B;++b)w.field[a][b]=weights[offset++];
#define LOAD1(field,A) for(int a=0;a<A;++a)w.field[a]=weights[offset++];
  LOAD2(wi,D,C) LOAD1(bi,D) LOAD2(wc,D,3) LOAD1(bc,D)
  LOAD1(gamma,D) LOAD1(beta,D) LOAD2(ws,2*S,D) LOAD1(bs,2*S)
  LOAD2(wb,D,S) LOAD1(bb,D) LOAD2(wo,C,D) LOAD1(bo,C)
  LOAD2(wh,C,S) LOAD1(bf,C)
#undef LOAD1
#undef LOAD2
#if MABA_VARIANT == 0
  batch<P>(input,w,output,0,P);
#else
  for(int base=0;base<P;base+=MABA_TILE) {
    int n=(base+MABA_TILE<=P)?MABA_TILE:P-base;
#if MABA_VARIANT == 1
    batch<MABA_TILE>(input,w,output,base,n);
#else
    streaming(input,w,output,base,n);
#endif
  }
#endif
}
