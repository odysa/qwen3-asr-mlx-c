#include "qwen_asr.h"
#include "audio.h"
#include "tokenizer.h"
#ifdef QWEN_WITH_MLX_C
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/transforms.h"
#include "mlx/c/half.h"
#include "mlx/c/error.h"
#endif
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum { QWEN_EOS = 151643, QWEN_IM_END = 151645, QWEN_MAX_NEW_TOKENS = 256 };
static int mlx_error_seen = 0;
static void mlx_error_to_stderr(const char *msg, void *data) { (void)data; mlx_error_seen = 1; fprintf(stderr, "qwen3-asr: MLX error: %s\n", msg ? msg : "unknown"); }
static void usage(const char *p) { fprintf(stderr, "usage: %s transcribe --model DIR --audio FILE [--language auto|English|Chinese] [--json] [--dump-dir DIR]\n", p); }
static int mkdir_one(const char *p) { return mkdir(p, 0755) == 0 || errno == EEXIST ? 0 : -1; }
static int write_blob(const char *dir, const char *name, const void *p, size_t n) {
  char path[4096]; if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) return -1;
  FILE *f = fopen(path, "wb"); if (!f) return -1; int ok = fwrite(p, 1, n, f) == n; if (fclose(f) != 0) ok = 0; return ok ? 0 : -1;
}
#ifdef QWEN_WITH_MLX_C
static int dump_shape(const char *dir, const char *stem, const int *shape, size_t ndim, const char *dtype) {
  char path[4096]; if (snprintf(path, sizeof(path), "%s/%s.json", dir, stem) >= (int)sizeof(path)) return -1;
  FILE *f = fopen(path, "w"); if (!f) return -1; fprintf(f, "{\"dtype\":\"%s\",\"shape\":[", dtype);
  for (size_t i = 0; i < ndim; i++) fprintf(f, "%s%d", i ? "," : "", shape[i]); fprintf(f, "]}\n"); return fclose(f) == 0 ? 0 : -1;
}
static int dump_array_f32(const char *dir, const char *stem, mlx_array a) {
  if (!a.ctx || mlx_array_eval(a) != 0) return -1; size_t n = mlx_array_size(a); float *v = malloc((n ? n : 1) * sizeof(float)); if (!v) return -1;
  switch (mlx_array_dtype(a)) {
    case MLX_FLOAT32: memcpy(v, mlx_array_data_float32(a), n * sizeof(float)); break;
    case MLX_BFLOAT16: { const bfloat16_t *p = mlx_array_data_bfloat16(a); for (size_t i = 0; i < n; i++) v[i] = (float)p[i]; break; }
    case MLX_FLOAT16: { const float16_t *p = mlx_array_data_float16(a); for (size_t i = 0; i < n; i++) v[i] = (float)p[i]; break; }
    default: free(v); return -1;
  }
  char bin[128]; snprintf(bin, sizeof(bin), "%s.bin", stem); int rc = write_blob(dir, bin, v, n * sizeof(float)); free(v);
  if (rc == 0) rc = dump_shape(dir, stem, mlx_array_shape(a), mlx_array_ndim(a), "float32"); return rc;
}
static int dump_ids(const char *dir, const char *stem, const int64_t *ids, size_t n) {
  char bin[128]; snprintf(bin, sizeof(bin), "%s.bin", stem); int rc = write_blob(dir, bin, ids, n * sizeof(*ids)); int shape = (int)n;
  if (rc == 0) rc = dump_shape(dir, stem, &shape, 1, "int64"); return rc;
}
static int argmax_token(mlx_array logits, mlx_stream stream, uint32_t *out) {
  mlx_array index = mlx_array_new();
  if (mlx_argmax_axis(&index, logits, 1, false, stream) != 0 || mlx_array_eval(index) != 0) { mlx_array_free(index); return -1; }
  int rc = -1;
  switch (mlx_array_dtype(index)) {
    case MLX_UINT32: *out = mlx_array_data_uint32(index)[0]; rc = 0; break;
    case MLX_INT32: *out = (uint32_t)mlx_array_data_int32(index)[0]; rc = 0; break;
    case MLX_INT64: *out = (uint32_t)mlx_array_data_int64(index)[0]; rc = 0; break;
    default: break;
  }
  mlx_array_free(index); return rc;
}
static int argmax_token_array(mlx_array logits, mlx_stream stream, mlx_array *out) {
  *out = mlx_array_new();
  return mlx_argmax_axis(out, logits, 1, false, stream);
}
static int kick_token_array(mlx_array token) {
  mlx_vector_array v = mlx_vector_array_new_value(token);
  int rc = mlx_async_eval(v);
  mlx_vector_array_free(v);
  return rc;
}

#endif

int main(int argc, char **argv) {
  const char *model_path = NULL, *audio_path = NULL, *dump_dir = NULL, *language = NULL; int json = 0;
  if (argc < 2 || strcmp(argv[1], "transcribe")) { usage(argv[0]); return 2; }
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
    else if (!strcmp(argv[i], "--audio") && i + 1 < argc) audio_path = argv[++i];
    else if (!strcmp(argv[i], "--json")) json = 1;
    else if (!strcmp(argv[i], "--dump-dir") && i + 1 < argc) dump_dir = argv[++i];
    else if (!strcmp(argv[i], "--language") && i + 1 < argc) language = argv[++i];
    else { usage(argv[0]); return 2; }
  }
  if (!model_path || !audio_path) { usage(argv[0]); return 2; }
#ifndef QWEN_WITH_MLX_C
  fprintf(stderr, "qwen3-asr: built without MLX C support\n"); return 1;
#else
  mlx_set_error_handler(mlx_error_to_stderr, NULL, NULL);
  fprintf(stderr, "qwen3-asr: model=%s audio=%s backend=c11+mlx-c+metal\n", model_path, audio_path);
  if (dump_dir && mkdir_one(dump_dir) != 0) { fprintf(stderr, "qwen3-asr: cannot create dump-dir: %s\n", strerror(errno)); return 1; }
  char err[512] = {0}; qwen_model_t *model = NULL; qwen_model_opts_t opts = {.threads = 1};
  if (qwen_model_load(model_path, &opts, &model, err, sizeof(err)) != 0) { fprintf(stderr, "qwen3-asr: model load failed: %s\n", err[0] ? err : "unknown error"); return 1; }
  int samples_n = 0, frames = 0; float *samples = qwen_load_wav(audio_path, &samples_n); float *mel = samples ? qwen_mel_spectrogram(samples, samples_n, &frames) : NULL; free(samples);
  if (!mel || frames <= 0) { fprintf(stderr, "qwen3-asr: WAV must be 16 kHz mono PCM16\n"); free(mel); qwen_model_free(model); return 1; }
  int mel_shape[2] = {128, frames}; mlx_array mel_a = mlx_array_new_data(mel, mel_shape, 2, MLX_FLOAT32); free(mel);
  mlx_array audio = mlx_array_new(), logits = mlx_array_new();
  if (!mel_a.ctx || qwen_encode(model, mel_a, &audio, err, sizeof(err)) != 0) { fprintf(stderr, "qwen3-asr: encoder failed: %s\n", err); mlx_array_free(mel_a); qwen_model_free(model); return 1; }
  if (dump_dir && dump_array_f32(dump_dir, "case_mel", mel_a) != 0) { fprintf(stderr, "qwen3-asr: failed to write mel dump\n"); mlx_array_free(mel_a); qwen_model_free(model); return 1; } mlx_array_free(mel_a);
  int audio_tokens = (int)mlx_array_dim(audio, 0); char vocab_path[4096]; snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_path);
  qwen_tokenizer_t *tok = qwen_tokenizer_load(vocab_path); size_t prompt_n = 0;
  const char *prompt_language = (language && strcmp(language, "auto")) ? language : NULL;
  if (language && strcmp(language, "auto") && strcmp(language, "English") && strcmp(language, "Chinese")) { fprintf(stderr, "qwen3-asr: --language must be auto, English, or Chinese\n"); mlx_array_free(audio); qwen_model_free(model); qwen_tokenizer_free(tok); return 2; }
  int64_t *prompt = tok ? qwen_tokenizer_prompt(tok, (size_t)audio_tokens, prompt_language, &prompt_n) : NULL;
  if (!tok || !prompt) { fprintf(stderr, "qwen3-asr: tokenizer/prompt construction failed\n"); qwen_tokenizer_free(tok); mlx_array_free(audio); qwen_model_free(model); return 1; }
  if (dump_dir && (dump_array_f32(dump_dir, "case_encoder", audio) != 0 || dump_ids(dump_dir, "case_prompt_ids", prompt, prompt_n) != 0)) { fprintf(stderr, "qwen3-asr: failed to write encoder/prompt dump\n"); free(prompt); qwen_tokenizer_free(tok); mlx_array_free(audio); qwen_model_free(model); return 1; }
  if (qwen_prefill(model, prompt, prompt_n, audio, &logits, err, sizeof(err)) != 0) { fprintf(stderr, "qwen3-asr: prefill failed: %s\n", err); free(prompt); qwen_tokenizer_free(tok); mlx_array_free(audio); qwen_model_free(model); return 1; }
  if (dump_dir && dump_array_f32(dump_dir, "case_first_logits", logits) != 0) { fprintf(stderr, "qwen3-asr: failed to write logits dump\n"); mlx_array_free(logits); free(prompt); qwen_tokenizer_free(tok); mlx_array_free(audio); qwen_model_free(model); return 1; }
  mlx_stream stream = mlx_default_gpu_stream_new(); int64_t generated[QWEN_MAX_NEW_TOKENS]; size_t generated_n = 0;
  int decode_ok = 0, hit_eos = 0;
  mlx_array token_array = mlx_array_new();
  if (argmax_token_array(logits, stream, &token_array) != 0 || !token_array.ctx || kick_token_array(token_array) != 0) {
    fprintf(stderr, "qwen3-asr: argmax failed\n");
    mlx_array_free(token_array);
  } else {
    mlx_array_free(logits); logits = mlx_array_new();
    for (; generated_n < QWEN_MAX_NEW_TOKENS; generated_n++) {
      mlx_array next_logits = mlx_array_new(), next_token = mlx_array_new();
      if (qwen_step_token_array(model, token_array, &next_logits, err, sizeof(err)) != 0 ||
          argmax_token_array(next_logits, stream, &next_token) != 0 ||
          !next_token.ctx || kick_token_array(next_token) != 0) {
        fprintf(stderr, "qwen3-asr: asynchronous decoder step failed: %s\n", err[0] ? err : "unknown error");
        mlx_array_free(next_logits); mlx_array_free(next_token); mlx_array_free(token_array); token_array = mlx_array_new();
        break;
      }
      uint32_t token = 0;
      if (mlx_array_item_uint32(&token, token_array) != 0 || token >= 151936) {
        fprintf(stderr, "qwen3-asr: asynchronous token evaluation/range failed\n");
        mlx_array_free(next_logits); mlx_array_free(next_token); mlx_array_free(token_array); token_array = mlx_array_new();
        break;
      }
      mlx_array_free(token_array); token_array = mlx_array_new(); mlx_array_free(next_logits);
      if (token == QWEN_EOS || token == QWEN_IM_END) {
        hit_eos = 1; decode_ok = 1; mlx_array_free(next_token); break;
      }
      generated[generated_n] = (int64_t)token;
      token_array = next_token;
    }
    if (generated_n == QWEN_MAX_NEW_TOKENS) mlx_array_free(token_array);
  }
  /* async argmax/step work, including the EOS lookahead, must finish before
     releasing arrays, resetting KV state, or reporting success. */
  if (mlx_synchronize(stream) != 0) {
    fprintf(stderr, "qwen3-asr: final GPU synchronization failed\n");
    decode_ok = 0;
  }
  if (generated_n == QWEN_MAX_NEW_TOKENS && !hit_eos) decode_ok = 0;
  mlx_stream_free(stream); mlx_array_free(logits);
  if (qwen_reset(model, err, sizeof(err)) != 0) {
    fprintf(stderr, "qwen3-asr: reset failed: %s\n", err[0] ? err : "unknown error");
    decode_ok = 0;
  }
  int64_t filtered[QWEN_MAX_NEW_TOKENS]; size_t filtered_n = 0; size_t start = 0;
  if (!prompt_language) { for (size_t i = 0; i < generated_n; i++) if (generated[i] == 151704) { start = i + 1; break; } }
  for (size_t i = start; i < generated_n; i++) if (generated[i] < 151644 || generated[i] > 151711) filtered[filtered_n++] = generated[i];
  char *text = decode_ok ? qwen_tokenizer_decode_all(tok, filtered, filtered_n) : NULL;
  if (!text) { fprintf(stderr, "qwen3-asr: decode did not reach EOS or tokenizer failed\n"); free(prompt); qwen_tokenizer_free(tok); mlx_array_free(audio); qwen_model_free(model); return 1; }
  if (json) { fputs("{\"text\":\"", stdout); for (char *p = text; *p; p++) { unsigned char c = (unsigned char)*p; if (c == '\\' || c == '\"') printf("\\%c", c); else if (c == '\n') fputs("\\n", stdout); else if (c == '\r') fputs("\\r", stdout); else if (c == '\t') fputs("\\t", stdout); else if (c < 0x20) printf("\\u%04x", c); else fputc(c, stdout); } printf("\",\"tokens\":%zu,\"status\":\"%s\"}\n", filtered_n, hit_eos ? "ok" : "truncated"); } else printf("%s\n", text);
  free(text); free(prompt); qwen_tokenizer_free(tok); mlx_array_free(audio); qwen_model_free(model); return 0;
#endif
}
