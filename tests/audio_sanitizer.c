#include "audio.h"
#include "tokenizer.h"
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
int main(int argc,char **argv){
 assert(argc==3);int n=0,f=0;float *x=qwen_load_wav(argv[1],&n);assert(x&&n>0);float *mel=qwen_mel_spectrogram(x,n,&f);assert(mel&&f==n/160);free(mel);free(x);
 uint8_t bad[1024];for(int i=0;i<1024;i++)bad[i]=(uint8_t)(i*19);for(int i=0;i<1024;i++)assert(!qwen_parse_wav_buffer(bad,(size_t)i,&n));
 float shortx[160]={0};mel=qwen_mel_spectrogram(shortx,160,&f);assert(mel&&f==1);free(mel);shortx[0]=NAN;assert(!qwen_mel_spectrogram(shortx,160,&f));
 qwen_tokenizer_t *t=qwen_tokenizer_load(argv[2]);assert(t);size_t count=0;int64_t *ids=qwen_tokenizer_prompt(t,13,"English",&count);assert(ids&&count>13);free(ids);int64_t abc[]={44,34301,13};char *s=qwen_tokenizer_decode_all(t,abc,3);assert(s);free(s);int64_t badid[]={-1};assert(!qwen_tokenizer_decode_all(t,badid,1));qwen_tokenizer_free(t);return 0;
}
