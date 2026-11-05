#include "whisper.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/resource.h>
static uint16_t u16(const unsigned char*p){return uint16_t(p[0])|(uint16_t(p[1])<<8);}
static uint32_t u32(const unsigned char*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
static std::string escape(const std::string&s){std::string r; for(unsigned char c:s){if(c=='"'||c=='\\'){r+='\\';r+=c;}else if(c=='\n')r+="\\n";else if(c=='\r')r+="\\r";else if(c=='\t')r+="\\t";else if(c<32)r+=' ';else r+=c;}return r;}
static bool read_wav(const char*path,std::vector<float>&out){std::ifstream f(path,std::ios::binary);if(!f)return false;std::vector<unsigned char>b((std::istreambuf_iterator<char>(f)),{});if(b.size()<12||memcmp(b.data(),"RIFF",4)||memcmp(b.data()+8,"WAVE",4))return false;bool fmt=false;size_t off=12;while(off+8<=b.size()){auto*p=b.data()+off;uint32_t n=u32(p+4);off+=8;if(n>b.size()-off)return false;if(!memcmp(p,"fmt ",4)){if(n<16||u16(b.data()+off)!=1||u16(b.data()+off+2)!=1||u32(b.data()+off+4)!=16000||u16(b.data()+off+14)!=16)return false;fmt=true;}else if(!memcmp(p,"data",4)){if(!fmt||n%2)return false;for(size_t i=0;i<n;i+=2){int16_t v=(int16_t)u16(b.data()+off+i);out.push_back(v/32768.0f);}return !out.empty();}off+=n+(n&1);}return false;}
int main(int argc,char**argv){if(argc!=6){std::cerr<<"usage: model wav language warmups repeats\n";return 2;}int warm=std::stoi(argv[4]),repeat=std::stoi(argv[5]);if(warm<0||repeat<1)return 2;using clock=std::chrono::steady_clock;auto start=clock::now();auto cp=whisper_context_default_params();cp.use_gpu=true;auto*ctx=whisper_init_from_file_with_params(argv[1],cp);if(!ctx)return 1;auto load=std::chrono::duration<double>(clock::now()-start).count();std::vector<float>audio;if(!read_wav(argv[2],audio)){whisper_free(ctx);return 1;}std::cout<<"{\"load_s\":"<<load<<"}"<<std::endl;for(int i=0;i<warm+repeat;i++){auto p=whisper_full_default_params(WHISPER_SAMPLING_GREEDY);p.n_threads=4;p.language=argv[3];p.greedy.best_of=1;p.temperature=0;p.temperature_inc=0;p.no_context=true;p.print_progress=false;p.print_realtime=false;p.print_timestamps=false;p.print_special=false;auto a=clock::now();int rc=whisper_full(ctx,p,audio.data(),audio.size());std::string text;for(int j=0;j<whisper_full_n_segments(ctx);j++)text+=whisper_full_get_segment_text(ctx,j);double sec=std::chrono::duration<double>(clock::now()-a).count();struct rusage ru{};getrusage(RUSAGE_SELF,&ru);std::cout<<"{\"run\":"<<i<<",\"warmup\":"<<(i<warm?"true":"false")<<",\"total_s\":"<<std::setprecision(9)<<sec<<",\"status\":\""<<(rc?"error":"ok")<<"\",\"termination\":\""<<(rc?"error":"whisper_full_returned")<<"\",\"rss_peak_bytes\":"<<ru.ru_maxrss<<",\"text\":\""<<escape(text)<<"\"}"<<std::endl;if(rc){whisper_free(ctx);return 1;}}whisper_free(ctx);}
