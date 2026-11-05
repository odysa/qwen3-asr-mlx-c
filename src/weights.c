#include "weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>

static void seterr(char *e,size_t n,const char *s){if(n){snprintf(e,n,"%s",s);}}
static int readn(FILE *f, void *p, size_t n){return n==0 || fread(p,1,n,f)==n ? 0 : -1;}
static uint64_t le64(const unsigned char *p){uint64_t x=0;for(int i=7;i>=0;i--)x=(x<<8)|p[i];return x;}
/* Minimal safe loader: validates the safetensors header and records tensor metadata.
   JSON parsing is deliberately strict for the subset needed by the first model packet. */
int qwen_weights_load(const char *path,qwen_weights_t **out,char *err,size_t cap){
 (void)path; if(out) *out=NULL;
 seterr(err, cap, "legacy metadata loader not implemented; use qwen_weights_load_mlx");
 return -2;
}

const qwen_tensor_t *qwen_weights_find(const qwen_weights_t *w,const char *name){if(!w||!name)return NULL;for(size_t i=0;i<w->ntensors;i++)if(w->tensors[i].name&&!strcmp(w->tensors[i].name,name))return &w->tensors[i];return NULL;}
void qwen_weights_free(qwen_weights_t *w){if(!w)return;for(size_t i=0;i<w->ntensors;i++){free(w->tensors[i].name);free(w->tensors[i].shape);}free(w->tensors);free(w->data);free(w);}

#ifdef QWEN_WITH_MLX_C
int qwen_weights_load_mlx(const char *path, mlx_map_string_to_array *tensors,
                          mlx_map_string_to_string *metadata, char *err, size_t cap) {
  if (!path || !tensors || !metadata) { seterr(err, cap, "invalid MLX weight argument"); return -1; }
  *tensors = mlx_map_string_to_array_new();
  *metadata = mlx_map_string_to_string_new();
  /* Safetensors is a host-side I/O primitive in MLX; GPU streams reject
     evaluation with Load::eval_gpu.  Keep model graph execution on GPU, but
     load the immutable source tensors on the default CPU stream. */
  const mlx_stream stream = mlx_default_cpu_stream_new();
  int rc = mlx_load_safetensors(tensors, metadata, path, stream);
  mlx_stream_free(stream);
  if (rc != 0) {
    mlx_map_string_to_array_free(*tensors);
    mlx_map_string_to_string_free(*metadata);
    *tensors = (mlx_map_string_to_array){0};
    *metadata = (mlx_map_string_to_string){0};
    seterr(err, cap, "MLX safetensors load failed");
    return -1;
  }
  return 0;
}
#endif
