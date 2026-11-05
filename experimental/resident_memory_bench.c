#include "qwen_asr.h"
#include "audio.h"
#include "tokenizer.h"
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/transforms.h"
#include "mlx/c/memory.h"
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + (double)t.tv_nsec / 1e9; }
static int argmax_array(mlx_array a, mlx_stream s, mlx_array *out) { *out = mlx_array_new(); return mlx_argmax_axis(out, a, 1, false, s); }
static int kick(mlx_array token) { mlx_vector_array v = mlx_vector_array_new_value(token); int rc = mlx_async_eval(v); mlx_vector_array_free(v); return rc; }
static void print_json(const char *s) {
  if (!s) return;
  for (; *s; s++) { unsigned char c = (unsigned char)*s; if (c == '\\' || c == '"') printf("\\%c", c); else if (c == '\n') printf("\\n"); else if (c == '\r') printf("\\r"); else if (c == '\t') printf("\\t"); else if (c < 0x20) printf("\\u%04x", c); else putchar(c); }
}

int main(int ac, char **av) {
  const char *m = NULL, *wav = NULL, *lang = "Chinese"; int warm = 1, repeat = 5;
  for (int i = 1; i < ac; i++) {
    if (!strcmp(av[i], "--model") && i + 1 < ac) m = av[++i];
    else if (!strcmp(av[i], "--audio") && i + 1 < ac) wav = av[++i];
    else if (!strcmp(av[i], "--language") && i + 1 < ac) lang = av[++i];
    else if (!strcmp(av[i], "--warmups") && i + 1 < ac) warm = atoi(av[++i]);
    else if (!strcmp(av[i], "--repeat") && i + 1 < ac) repeat = atoi(av[++i]);
    else { fprintf(stderr, "usage: %s --model DIR --audio WAV [--language Chinese|English] [--warmups N] [--repeat N]\n", av[0]); return 2; }
  }
  if (!m || !wav || warm < 0 || repeat < 1) return 2;
  char e[256] = {0}; qwen_model_t *mod = NULL; qwen_model_opts_t o = {1};
  double t0 = now_s(); if (qwen_model_load(m, &o, &mod, e, sizeof(e))) { fprintf(stderr, "load: %s\n", e); return 1; }
  double load = now_s() - t0; int ns = 0; float *s = qwen_load_wav(wav, &ns);
  if (!s) { fprintf(stderr, "bad wav\n"); qwen_model_free(mod); return 1; }
  char vp[4096]; snprintf(vp, sizeof(vp), "%s/vocab.json", m); qwen_tokenizer_t *tok = qwen_tokenizer_load(vp);
  if (!tok) { qwen_model_free(mod); free(s); return 1; }
  mlx_stream gs = mlx_default_gpu_stream_new();
  size_t setup_active=0,setup_peak=0;struct rusage setup_usage;
  if(mlx_get_active_memory(&setup_active)||mlx_get_peak_memory(&setup_peak)||getrusage(RUSAGE_SELF,&setup_usage))return 1;
  printf("{\"load_s\":%.6f,\"resident_setup_s\":%.6f,\"mlx_active_after_setup_bytes\":%zu,\"mlx_peak_after_setup_bytes\":%zu,\"rss_peak_after_setup_bytes\":%llu,\"warmups\":%d,\"repeat\":%d}\n",load,now_s()-t0,setup_active,setup_peak,(unsigned long long)setup_usage.ru_maxrss,warm,repeat);
  int total = warm + repeat;
  for (int run = 0; run < total; run++) {
    double a = now_s(); int nf = 0; float *mel = qwen_mel_spectrogram(s, ns, &nf); double af = now_s();
    int sh[2] = {128, nf}; mlx_array ma = mel ? mlx_array_new_data(mel, sh, 2, MLX_FLOAT32) : (mlx_array){0}; free(mel);
    mlx_array enc = mlx_array_new(), logits = mlx_array_new();
    if (!ma.ctx || qwen_encode(mod, ma, &enc, e, sizeof(e)) || mlx_synchronize(gs) != 0) { fprintf(stderr, "encode/synchronize: %s\n", e); return 1; }
    double b = now_s(); size_t pn = 0; int64_t *p = qwen_tokenizer_prompt(tok, mlx_array_dim(enc, 0), lang, &pn);
    if (!p || qwen_prefill(mod, p, pn, enc, &logits, e, sizeof(e)) || mlx_synchronize(gs) != 0) { fprintf(stderr, "prefill/synchronize: %s\n", e); return 1; }
    double c = now_s(); int64_t ids[256]; size_t n = 0; int eos = 0; int ok = 1; mlx_array y = mlx_array_new();
    if (argmax_array(logits, gs, &y) || !y.ctx || kick(y)) ok = 0;
    mlx_array_free(logits); logits = mlx_array_new();
    for (; ok && n < 256; n++) {
      mlx_array next_logits = mlx_array_new(), next_y = mlx_array_new();
      if (qwen_step_token_array(mod, y, &next_logits, e, sizeof(e)) || argmax_array(next_logits, gs, &next_y) || !next_y.ctx || kick(next_y)) { fprintf(stderr, "step/argmax: %s\n", e); mlx_array_free(next_logits); mlx_array_free(next_y); ok = 0; break; }
      uint32_t id = 0;
      if (mlx_array_item_uint32(&id, y) != 0 || id >= 151936) { fprintf(stderr, "token evaluation/range failed\n"); mlx_array_free(next_logits); mlx_array_free(next_y); ok = 0; break; }
      mlx_array_free(y); y = mlx_array_new(); mlx_array_free(next_logits);
      if (id == 151643 || id == 151645) { eos = 1; mlx_array_free(next_y); break; }
      ids[n] = (int64_t)id; y = next_y;
    }
    if (n == 256 && ok) { fprintf(stderr, "decode truncated\n"); ok = 0; }
    if (y.ctx) mlx_array_free(y);
    if (mlx_synchronize(gs) != 0) { fprintf(stderr, "GPU synchronization failed\n"); ok = 0; }
    double d_gpu = now_s(); char *txt = ok && eos ? qwen_tokenizer_decode_all(tok, ids, n) : NULL; double d = now_s();
    if (!txt) { fprintf(stderr, "decode failed or missing EOS\n"); return 1; }
    size_t active_bytes=0,peak_bytes=0,cache_bytes=0;struct rusage usage;
    if(mlx_get_active_memory(&active_bytes)||mlx_get_peak_memory(&peak_bytes)||mlx_get_cache_memory(&cache_bytes)||getrusage(RUSAGE_SELF,&usage))return 1;
    printf("{\"run\":%d,\"warmup\":%s,\"frontend_s\":%.6f,\"encode_s\":%.6f,\"prefill_s\":%.6f,\"decode_s\":%.6f,\"total_s\":%.6f,\"tokens\":%zu,\"token_ids\":[", run, run < warm ? "true" : "false", af-a, b-af, c-b, d_gpu-c, d-a, n);
    for (size_t k = 0; k < n; k++) printf("%s%lld", k ? "," : "", (long long)ids[k]); printf("],\"text\":\""); print_json(txt); printf("\",\"eos\":true,\"mlx_active_bytes\":%zu,\"mlx_peak_bytes\":%zu,\"mlx_cache_bytes\":%zu,\"rss_peak_bytes\":%llu}\n",active_bytes,peak_bytes,cache_bytes,(unsigned long long)usage.ru_maxrss);
    free(txt); mlx_array_free(ma); mlx_array_free(enc); mlx_array_free(logits); free(p); if (qwen_reset(mod, e, sizeof(e)) != 0) { fprintf(stderr, "reset failed: %s\n", e); return 1; }
  }
  mlx_stream_free(gs); qwen_tokenizer_free(tok); free(s); qwen_model_free(mod); return 0;
}
