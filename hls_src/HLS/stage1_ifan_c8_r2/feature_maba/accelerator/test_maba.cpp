#include "maba_top.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
using namespace maba_hw;
static float x[COUNT],y[COUNT],ref[COUNT],weights[WCOUNT];
bool read(const std::string &p,float *v,int n) {
 std::ifstream f(p.c_str(),std::ios::binary);
 return bool(f.read(reinterpret_cast<char*>(v),n*sizeof(float))) && f.peek()==EOF;
}
int main(int argc,char **argv) {
 if(argc<2){std::cerr<<"usage: test_maba DATA_DIR [limit] [output_dir]\n";return 2;}
 std::string dir=argv[1];int limit=argc>2?std::stoi(argv[2]):0;
 if(!read(dir+"/weights.bin",weights,WCOUNT))return 3;
 std::ifstream list((dir+"/cases.txt").c_str());
 if(!list)return 4;
 std::string name;int valid,count=0;bool all=true;
 while(list>>name>>valid) {
  if(limit && count>=limit)break;
  std::ifstream special((dir+"/"+name+".weights.bin").c_str(),std::ios::binary);
  if(!read(special.good()?dir+"/"+name+".weights.bin":dir+"/weights.bin",weights,WCOUNT))return 8;
  if(!read(dir+"/"+name+".input.bin",x,COUNT)||!read(dir+"/"+name+".output.bin",ref,COUNT))return 5;
  maba_top(x,weights,y);
  double mx=0,sq=0;bool finite=true;
  for(int i=0;i<valid*C*P;++i){finite=finite&&std::isfinite(y[i])&&std::isfinite(ref[i]);double e=double(y[i])-ref[i];mx=std::fmax(mx,std::fabs(e));sq+=e*e;}
  double rmse=std::sqrt(sq/(valid*C*P));bool ok=finite&&mx<=1e-5&&rmse<=1e-6;all=all&&ok;
  std::cout<<std::setprecision(10)<<"CASE "<<name<<" max_abs="<<mx<<" rmse="<<rmse<<" pass="<<ok<<std::endl;
  if(argc>3){std::ofstream f((std::string(argv[3])+"/"+name+".bin").c_str(),std::ios::binary);f.write(reinterpret_cast<char*>(y),sizeof(y));if(!f)return 6;}
  ++count;
 }
 if(!count)return 7;
 std::cout<<"SUMMARY cases="<<count<<" variant="<<MABA_VARIANT<<" tile="<<MABA_TILE<<" par="<<MABA_PAR<<" pass="<<all<<"\n";
 return all?0:1;
}
