/* Resident subprocess bridge for the local HTTP service.
 * Reuses the shipped MLX C graph, tokenizer and asynchronous greedy decoder.
 * Private pipe protocol: LANGUAGE<TAB>WAV_PATH<LF>; JSON response per line.
 */
#include "qwen_asr.h"
#include "audio.h"
#include "tokenizer.h"
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/transforms.h"
#include "mlx/c/error.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static int error_seen=0;
static void on_error(const char *msg,void *data) { (void)data;error_seen=1;fprintf(stderr,"worker MLX error: %s\n",msg?msg:""); }
static void json_text(const char *s) {
  for(;*s;s++){unsigned char c=(unsigned char)*s;
    if(c=='\\'||c=='"')printf("\\%c",c);
    else if(c<32)printf("\\u%04x",c);else putchar(c);
  }
}
static int argmax(mlx_array logits,mlx_stream stream,mlx_array *out){*out=mlx_array_new();return mlx_argmax_axis(out,logits,1,false,stream);}
static int kick(mlx_array token){mlx_vector_array v=mlx_vector_array_new_value(token);int rc=mlx_async_eval(v);mlx_vector_array_free(v);return rc;}

static int transcribe(qwen_model_t *model,qwen_tokenizer_t *tok,mlx_stream stream,const char *path,const char *language){
  char error[512]={0};int ns=0,nf=0;double start=now_s();
  float *samples=qwen_load_wav(path,&ns);float *mel=samples?qwen_mel_spectrogram(samples,ns,&nf):NULL;free(samples);
  if(!mel||nf<=0){free(mel);puts("{\"error\":\"invalid WAV\"}");fflush(stdout);return 0;}
  int shape[2]={128,nf};mlx_array ma=mlx_array_new_data(mel,shape,2,MLX_FLOAT32);free(mel);
  mlx_array enc=mlx_array_new(),logits=mlx_array_new(),y=mlx_array_new();int64_t *prompt=NULL;size_t pn=0,n=0;int eos=0,ok=1;
  if(!ma.ctx||qwen_encode(model,ma,&enc,error,sizeof(error))||mlx_synchronize(stream))ok=0;
  double encoded=now_s();
  const char *forced=strcmp(language,"auto")?language:NULL;
  if(ok){prompt=qwen_tokenizer_prompt(tok,mlx_array_dim(enc,0),forced,&pn);if(!prompt||qwen_prefill(model,prompt,pn,enc,&logits,error,sizeof(error))||mlx_synchronize(stream))ok=0;}
  double prefilled=now_s();
  if(ok&&(argmax(logits,stream,&y)||!y.ctx||kick(y)))ok=0;
  mlx_array_free(logits);logits=mlx_array_new();int64_t ids[256];
  for(;ok&&n<256;n++){
    mlx_array next_logits=mlx_array_new(),next_y=mlx_array_new();
    if(qwen_step_token_array(model,y,&next_logits,error,sizeof(error))||argmax(next_logits,stream,&next_y)||!next_y.ctx||kick(next_y)){
      mlx_array_free(next_logits);mlx_array_free(next_y);ok=0;break;
    }
    uint32_t id=0;
    if(mlx_array_item_uint32(&id,y)||id>=151936){mlx_array_free(next_logits);mlx_array_free(next_y);ok=0;break;}
    mlx_array_free(y);y=mlx_array_new();mlx_array_free(next_logits);
    if(id==151643||id==151645){eos=1;mlx_array_free(next_y);break;}
    ids[n]=id;y=next_y;
  }
  mlx_array_free(y);if(mlx_synchronize(stream)||error_seen)ok=0;
  /* Reset only after all lookahead GPU work has completed. */
  if(qwen_reset(model,error,sizeof(error)))ok=0;
  int64_t filtered[256];size_t fn=0,offset=0;
  if(!forced){for(size_t i=0;i<n;i++)if(ids[i]==151704){offset=i+1;break;}}
  for(size_t i=offset;i<n;i++)if(ids[i]<151644||ids[i]>151711)filtered[fn++]=ids[i];
  char *text=ok&&eos?qwen_tokenizer_decode_all(tok,filtered,fn):NULL;
  mlx_array_free(ma);mlx_array_free(enc);mlx_array_free(logits);free(prompt);
  if(!text){fprintf(stderr,"worker failed or capped: %s\n",error);puts("{\"error\":\"native decode failed or reached token cap\"}");fflush(stdout);return 1;}
  struct rusage usage;getrusage(RUSAGE_SELF,&usage);
  printf("{\"text\":\"");json_text(text);
  printf("\",\"termination\":\"eos\",\"eos\":true,\"generation_tokens\":%zu,\"token_ids\":[",n);
  for(size_t i=0;i<n;i++)printf("%s%lld",i?",":"",(long long)ids[i]);
  printf("],\"native_total_s\":%.9f,\"encode_frontend_s\":%.9f,\"prefill_s\":%.9f,\"decode_materialize_s\":%.9f,\"native_peak_rss_bytes\":%llu}\n",now_s()-start,encoded-start,prefilled-encoded,now_s()-prefilled,(unsigned long long)usage.ru_maxrss);
  fflush(stdout);free(text);return 0;
}
int main(int argc,char **argv){
  if(argc!=3||strcmp(argv[1],"--model")){fprintf(stderr,"usage: %s --model DIR\n",argv[0]);return 2;}
  qwen_model_t *model=NULL;qwen_model_opts_t options={.threads=1};char error[512]={0};double start=now_s();
  mlx_set_error_handler(on_error,NULL,NULL);
  if(qwen_model_load(argv[2],&options,&model,error,sizeof(error))){fprintf(stderr,"worker load: %s\n",error);return 1;}
  char vocab[4096];if(snprintf(vocab,sizeof(vocab),"%s/vocab.json",argv[2])>=(int)sizeof(vocab))return 1;
  qwen_tokenizer_t *tok=qwen_tokenizer_load(vocab);if(!tok){qwen_model_free(model);return 1;}
  mlx_stream stream=mlx_default_gpu_stream_new();
  printf("{\"status\":\"ready\",\"load_prepare_s\":%.9f,\"weights_may_be_lazy\":true}\n",now_s()-start);fflush(stdout);
  char *line=NULL;size_t capacity=0;int exit_code=0;
  while(getline(&line,&capacity,stdin)>0){
    if(strlen(line)>8192){exit_code=2;break;}
    line[strcspn(line,"\r\n")]=0;char *tab=strchr(line,'\t');
    if(!tab||strchr(tab+1,'\t')){puts("{\"error\":\"invalid pipe request\"}");fflush(stdout);continue;}
    *tab++=0;
    if(strcmp(line,"English")&&strcmp(line,"Chinese")&&strcmp(line,"auto")){puts("{\"error\":\"unsupported language\"}");fflush(stdout);continue;}
    error_seen=0;if(transcribe(model,tok,stream,tab,line)){exit_code=1;break;}
  }
  free(line);mlx_stream_free(stream);qwen_tokenizer_free(tok);qwen_model_free(model);return exit_code;
}
