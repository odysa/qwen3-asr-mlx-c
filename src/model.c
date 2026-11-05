/* Qwen3-ASR 0.6B and 1.7B graph in C over Apple's MLX C API.
 * Architecture follows QwenLM/Qwen3-ASR and MLX-Audio (see review/upstream).
 * No host-side tensor arithmetic in the model's hot path. */
#include "mlx/c/compile.h"
#include "mlx/c/error.h"
#include "mlx/c/fast.h"
#include "mlx/c/ops.h"
#include "mlx/c/optional.h"
#include "qwen_asr.h"
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct qwen_model {
  mlx_map_string_to_array w;
  mlx_map_string_to_string metadata;
  mlx_stream s;
  mlx_array k[28], v[28];
  int offset;
  int hidden, enc_width, enc_heads, enc_layers;
  mlx_closure silu_compiled, gelu_compiled;
};
typedef struct {
  qwen_model_t *m;
  mlx_array *a;
  size_t n, cap;
  void **hosts;
  size_t nh, ch;
  jmp_buf jump;
  const char *why;
} arena;
#define TRY(a, expr)                                                           \
  do {                                                                         \
    if ((expr) != 0) {                                                         \
      (a)->why = #expr;                                                        \
      longjmp((a)->jump, 1);                                                   \
    }                                                                          \
  } while (0)
static mlx_array keep(arena *a, mlx_array x) {
  if (a->n == a->cap) {
    size_t c = a->cap ? a->cap * 2 : 256;
    mlx_array *p = realloc(a->a, c * sizeof(*p));
    if (!p) {
      mlx_array_free(x);
      a->why = "allocation failure";
      longjmp(a->jump, 1);
    }
    a->a = p;
    a->cap = c;
  }
  a->a[a->n++] = x;
  return x;
}
static mlx_array fresh(arena *a) { return keep(a, mlx_array_new_float32(0)); }
static void *host(arena *a, size_t bytes) {
  void *p = calloc(1, bytes);
  if (!p) {
    a->why = "host allocation";
    longjmp(a->jump, 1);
  }
  if (a->nh == a->ch) {
    size_t c = a->ch ? a->ch * 2 : 16;
    void **h = realloc(a->hosts, c * sizeof(*h));
    if (!h) {
      free(p);
      a->why = "host allocation registry";
      longjmp(a->jump, 1);
    }
    a->hosts = h;
    a->ch = c;
  }
  a->hosts[a->nh++] = p;
  return p;
}
static void clear(arena *a) {
  for (size_t i = a->n; i; i--)
    mlx_array_free(a->a[i - 1]);
  free(a->a);
  for (size_t i = 0; i < a->nh; i++)
    free(a->hosts[i]);
  free(a->hosts);
}
static _Thread_local char mlx_last_error[1024];
static void returning_error(const char *msg, void *data) {
  (void)data;
  snprintf(mlx_last_error, sizeof mlx_last_error, "%s",
           msg ? msg : "unknown MLX error");
}
static int failure(arena *a, char *e, size_t n) {
  if (e && n)
    snprintf(e, n, "MLX model operation failed: %s",
             mlx_last_error[0] ? mlx_last_error
                               : (a->why ? a->why : "unknown"));
  clear(a);
  free(a);
  return -1;
}
static mlx_array data(arena *a, const void *p, const int *shape, int n,
                      mlx_dtype d) {
  return keep(a, mlx_array_new_data(p, shape, n, d));
}
static mlx_array scalar(arena *a, float x) {
  return keep(a, mlx_array_new_float32(x));
}
static mlx_array cast(arena *a, mlx_array x, mlx_dtype d) {
  mlx_array y = fresh(a);
  TRY(a, mlx_astype(&y, x, d, a->m->s));
  return y;
}
static mlx_array weight(arena *a, const char *key) {
  mlx_array y = fresh(a);
  TRY(a, mlx_map_string_to_array_get(&y, a->m->w, key));
  return y;
}
static mlx_array named(arena *a, const char *prefix, const char *suffix) {
  char k[256];
  snprintf(k, sizeof k, "%s.%s", prefix, suffix);
  return weight(a, k);
}
static mlx_array reshape(arena *a, mlx_array x, const int *shape, int n) {
  mlx_array y = fresh(a);
  TRY(a, mlx_reshape(&y, x, shape, n, a->m->s));
  return y;
}
static mlx_array permute(arena *a, mlx_array x, const int *axes, int n) {
  mlx_array y = fresh(a);
  TRY(a, mlx_transpose_axes(&y, x, axes, n, a->m->s));
  return y;
}
static mlx_array add(arena *a, mlx_array x, mlx_array b) {
  mlx_array y = fresh(a);
  TRY(a, mlx_add(&y, x, b, a->m->s));
  return y;
}
static mlx_array mul(arena *a, mlx_array x, mlx_array b) {
  mlx_array y = fresh(a);
  TRY(a, mlx_multiply(&y, x, b, a->m->s));
  return y;
}
static mlx_array scale(arena *a, mlx_array x, float f) {
  return mul(a, x, cast(a, scalar(a, f), mlx_array_dtype(x)));
}
static mlx_array mm(arena *a, mlx_array x, mlx_array b) {
  mlx_array y = fresh(a);
  TRY(a, mlx_matmul(&y, x, b, a->m->s));
  return y;
}
static mlx_array linear(arena *a, mlx_array x, const char *name, int bias) {
  mlx_array w = named(a, name, "weight");
  mlx_array y = mm(a, x, permute(a, w, (int[]){1, 0}, 2));
  return bias ? add(a, y, named(a, name, "bias")) : y;
}
static mlx_array ln(arena *a, mlx_array x, const char *name) {
  mlx_array y = fresh(a);
  TRY(a, mlx_fast_layer_norm(&y, x, named(a, name, "weight"),
                             named(a, name, "bias"), 1e-5f, a->m->s));
  return y;
}
static mlx_array rms(arena *a, mlx_array x, const char *name) {
  mlx_array y = fresh(a);
  TRY(a, mlx_fast_rms_norm(&y, x, named(a, name, "weight"), 1e-6f, a->m->s));
  return y;
}
static int gelu_graph(mlx_array *out, const mlx_array x) {
  mlx_stream s = mlx_default_gpu_stream_new();
  mlx_array half = mlx_array_new_float32((float)sqrt(2.)),
            one = mlx_array_new_float32(1), two = mlx_array_new_float32(2),
            z = mlx_array_new(), e = mlx_array_new(), b = mlx_array_new(),
            p = mlx_array_new();
  int rc = mlx_divide(&z, x, half, s);
  if (!rc)
    rc = mlx_erf(&e, z, s);
  if (!rc)
    rc = mlx_add(&b, one, e, s);
  if (!rc)
    rc = mlx_multiply(&p, x, b, s);
  if (!rc)
    rc = mlx_divide(out, p, two, s);
  mlx_array_free(half);
  mlx_array_free(one);
  mlx_array_free(two);
  mlx_array_free(z);
  mlx_array_free(e);
  mlx_array_free(b);
  mlx_array_free(p);
  mlx_stream_free(s);
  return rc;
}
static mlx_array gelu(arena *a, mlx_array x) {
  mlx_array y = fresh(a);
  mlx_vector_array in = mlx_vector_array_new_value(x),
                   out = mlx_vector_array_new();
  int rc = mlx_closure_apply(&out, a->m->gelu_compiled, in);
  mlx_vector_array_free(in);
  if (!rc)
    rc = mlx_vector_array_get(&y, out, 0);
  mlx_vector_array_free(out);
  TRY(a, rc);
  return y;
}
static int silu_graph(mlx_array *out, const mlx_array x) {
  mlx_stream s = mlx_default_gpu_stream_new();
  mlx_array sig = mlx_array_new();
  int rc = mlx_sigmoid(&sig, x, s);
  if (!rc)
    rc = mlx_multiply(out, x, sig, s);
  mlx_array_free(sig);
  mlx_stream_free(s);
  return rc;
}
static mlx_array silu(arena *a, mlx_array x) {
  mlx_array y = fresh(a);
  mlx_vector_array in = mlx_vector_array_new_value(x),
                   out = mlx_vector_array_new();
  int rc = mlx_closure_apply(&out, a->m->silu_compiled, in);
  mlx_vector_array_free(in);
  if (!rc)
    rc = mlx_vector_array_get(&y, out, 0);
  mlx_vector_array_free(out);
  TRY(a, rc);
  return y;
}
static mlx_array slice(arena *a, mlx_array x, const int *start, const int *stop,
                       int n) {
  int strides[4] = {1, 1, 1, 1};
  mlx_array y = fresh(a);
  TRY(a, mlx_slice(&y, x, start, n, stop, n, strides, n, a->m->s));
  return y;
}
static mlx_array join(arena *a, const mlx_array *xs, size_t n, int axis,
                      int stack) {
  mlx_array y = fresh(a);
  mlx_vector_array v = mlx_vector_array_new_data(xs, n);
  int rc = stack ? mlx_stack_axis(&y, v, axis, a->m->s)
                 : mlx_concatenate_axis(&y, v, axis, a->m->s);
  mlx_vector_array_free(v);
  TRY(a, rc);
  return y;
}
static mlx_array sdpa(arena *a, mlx_array q, mlx_array k, mlx_array v,
                      float scale_, const char *mode, mlx_array mask) {
  mlx_array y = fresh(a);
  TRY(a, mlx_fast_scaled_dot_product_attention(&y, q, k, v, scale_, mode, mask,
                                               (mlx_array){0}, false, a->m->s));
  return y;
}
static mlx_array heads(arena *a, mlx_array x, int t, int nh, int dh) {
  return permute(a, reshape(a, x, (int[]){1, t, nh, dh}, 4),
                 (int[]){0, 2, 1, 3}, 4);
}
static mlx_array unheads(arena *a, mlx_array x, int t, int width) {
  return reshape(a, permute(a, x, (int[]){0, 2, 1, 3}, 4), (int[]){1, t, width},
                 3);
}
static mlx_array rope(arena *a, mlx_array x, int offset) {
  mlx_array y = fresh(a);
  TRY(a, mlx_fast_rope(&y, x, 128, false, (mlx_optional_float){1000000.f, true},
                       1.f, offset, (mlx_array){0}, a->m->s));
  return y;
}
static mlx_array enc_layer(arena *a, mlx_array x, int i, mlx_array mask) {
  char p[160], n[200];
  snprintf(p, sizeof p, "thinker.audio_tower.layers.%d", i);
  snprintf(n, sizeof n, "%s.self_attn_layer_norm", p);
  mlx_array h = ln(a, x, n);
  snprintf(n, sizeof n, "%s.self_attn.q_proj", p);
  mlx_array q = scale(a, linear(a, h, n, 1), .125f);
  snprintf(n, sizeof n, "%s.self_attn.k_proj", p);
  mlx_array k = linear(a, h, n, 1);
  snprintf(n, sizeof n, "%s.self_attn.v_proj", p);
  mlx_array v = linear(a, h, n, 1);
  int t = mlx_array_dim(x, 1);
  h = unheads(a,
              sdpa(a, heads(a, q, t, a->m->enc_heads, 64), heads(a, k, t, a->m->enc_heads, 64),
                   heads(a, v, t, a->m->enc_heads, 64), 1.f, "", mask),
              t, a->m->enc_width);
  snprintf(n, sizeof n, "%s.self_attn.out_proj", p);
  x = add(a, x, linear(a, h, n, 1));
  snprintf(n, sizeof n, "%s.final_layer_norm", p);
  h = ln(a, x, n);
  snprintf(n, sizeof n, "%s.fc1", p);
  h = gelu(a, linear(a, h, n, 1));
  snprintf(n, sizeof n, "%s.fc2", p);
  return add(a, x, linear(a, h, n, 1));
}

int qwen_model_load(const char *path, const qwen_model_opts_t *opts,
                    qwen_model_t **out, char *err, size_t cap) {
  (void)opts;
  mlx_set_error_handler(returning_error, NULL, NULL);
  mlx_last_error[0] = 0;
  if (!path || !out || strlen(path) > 4000)
    return -1;
  *out = NULL;
  qwen_model_t *m = calloc(1, sizeof(*m));
  if (!m)
    return -1;
  m->s = mlx_default_gpu_stream_new();
  char file[4096];
  size_t l = strlen(path);
  if (l > 12 && !strcmp(path + l - 12, ".safetensors"))
    snprintf(file, sizeof file, "%s", path);
  else
    {
      snprintf(file, sizeof file, "%s/model.safetensors", path);
      FILE *probe = fopen(file, "rb");
      if (probe) fclose(probe);
      else snprintf(file, sizeof file, "%s/model-00001-of-00002.safetensors", path);
    }
  if (qwen_weights_load_mlx(file, &m->w, &m->metadata, err, cap)) {
    mlx_stream_free(m->s);
    free(m);
    return -1;
  }
  /* Official 1.7B weights have two safetensors shards. */
  if (!(l > 12 && !strcmp(path + l - 12, ".safetensors"))) {
    char second[4096];
    snprintf(second, sizeof second, "%s/model-00002-of-00002.safetensors", path);
    if (strstr(file, "model-00001-of-00002.safetensors")) {
      mlx_map_string_to_array extra = {0};
      mlx_map_string_to_string meta = {0};
      if (qwen_weights_load_mlx(second, &extra, &meta, err, cap)) {
        qwen_model_free(m); return -1;
      }
      mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(extra);
      const char *key = NULL; mlx_array value = mlx_array_new();
      int failed = 0, next;
      while ((next = mlx_map_string_to_array_iterator_next(&key, &value, it)) == 0) {
        if (mlx_map_string_to_array_insert(m->w, key, value)) { failed = 1; break; }
      }
      if (!failed && next != 2) failed = 1;
      mlx_array_free(value); mlx_map_string_to_array_iterator_free(it);
      mlx_map_string_to_array_free(extra); mlx_map_string_to_string_free(meta);
      if (failed) { if (err && cap) snprintf(err, cap, "failed to merge model shards"); qwen_model_free(m); return -1; }
    }
  }
  mlx_array w = mlx_array_new();
  int bad = mlx_map_string_to_array_get(&w, m->w,
                                        "thinker.model.embed_tokens.weight") ||
            mlx_array_ndim(w) != 2 || mlx_array_dim(w, 0) != 151936;
  int hidden = bad ? 0 : mlx_array_dim(w, 1);
  mlx_array_free(w);
  if (hidden == 1024) { m->hidden = 1024; m->enc_width = 896; m->enc_heads = 14; m->enc_layers = 18; }
  else if (hidden == 2048) { m->hidden = 2048; m->enc_width = 1024; m->enc_heads = 16; m->enc_layers = 24; }
  else {
    if (err && cap) snprintf(err, cap, "expected Qwen3-ASR 0.6B or 1.7B BF16 weights");
    qwen_model_free(m); return -1;
  }
  mlx_closure raw = mlx_closure_new_unary(silu_graph);
  m->silu_compiled = mlx_closure_new();
  int crc = mlx_compile(&m->silu_compiled, raw, true);
  mlx_closure_free(raw);
  if (crc) {
    qwen_model_free(m);
    return -1;
  }
  raw = mlx_closure_new_unary(gelu_graph);
  m->gelu_compiled = mlx_closure_new();
  crc = mlx_compile(&m->gelu_compiled, raw, true);
  mlx_closure_free(raw);
  if (crc) {
    qwen_model_free(m);
    return -1;
  }
  *out = m;
  return 0;
}
int qwen_reset(qwen_model_t *m, char *err, size_t cap) {
  (void)err;
  (void)cap;
  if (!m)
    return -1;
  for (int i = 0; i < 28; i++) {
    if (m->k[i].ctx)
      mlx_array_free(m->k[i]);
    if (m->v[i].ctx)
      mlx_array_free(m->v[i]);
    m->k[i] = (mlx_array){0};
    m->v[i] = (mlx_array){0};
  }
  m->offset = 0;
  return 0;
}
void qwen_model_free(qwen_model_t *m) {
  if (!m)
    return;
  qwen_reset(m, NULL, 0);
  mlx_map_string_to_array_free(m->w);
  mlx_map_string_to_string_free(m->metadata);
  mlx_stream_free(m->s);
  if (m->silu_compiled.ctx)
    mlx_closure_free(m->silu_compiled);
  if (m->gelu_compiled.ctx)
    mlx_closure_free(m->gelu_compiled);
  free(m);
}

int qwen_encode(qwen_model_t *m, qwen_array_t mel, qwen_array_t *out, char *err,
                size_t cap) {
  if (!m || !out || !mel.ctx)
    return -1;
  mlx_last_error[0] = 0;
  arena *a = calloc(1, sizeof(*a));
  if (!a)
    return -1;
  a->m = m;
  if (setjmp(a->jump))
    return failure(a, err, cap);
  int nd = mlx_array_ndim(mel);
  if (nd == 3 && mlx_array_dim(mel, 0) == 1 && mlx_array_dim(mel, 1) == 128)
    mel = reshape(a, mel, (int[]){128, mlx_array_dim(mel, 2)}, 2);
  if (mlx_array_ndim(mel) != 2 || mlx_array_dim(mel, 0) != 128) {
    a->why = "mel must be [128,F]";
    longjmp(a->jump, 1);
  }
  int f = mlx_array_dim(mel, 1);
  if (f <= 0 || f > 3000) {
    a->why = "mel frame length outside supported range";
    longjmp(a->jump, 1);
  }
  int nc = (f + 99) / 100, maxchunk = f < 100 ? f : 100,
      ct = (maxchunk + 7) / 8;
  mlx_array *chunks = host(a, (size_t)nc * sizeof(*chunks));
  if (!chunks) {
    a->why = "chunk allocation";
    longjmp(a->jump, 1);
  }
  for (int i = 0; i < nc; i++) {
    int len = f - i * 100;
    if (len > 100)
      len = 100;
    chunks[i] =
        slice(a, mel, (int[]){0, i * 100}, (int[]){128, i * 100 + len}, 2);
    if (len < maxchunk) {
      mlx_array padded = fresh(a);
      TRY(a,
          mlx_pad(&padded, chunks[i], (int[]){1}, 1, (int[]){0}, 1,
                  (int[]){maxchunk - len}, 1, scalar(a, 0), "constant", m->s));
      chunks[i] = padded;
    }
  }
  mlx_array x = join(a, chunks, (size_t)nc, 0, 1);
  x = reshape(a, x, (int[]){nc, 128, maxchunk, 1}, 4);
  for (int i = 1; i <= 3; i++) {
    char n[100];
    snprintf(n, sizeof n, "thinker.audio_tower.conv2d%d", i);
    mlx_array w = permute(a, named(a, n, "weight"), (int[]){0, 2, 3, 1}, 4),
              y = fresh(a);
    TRY(a, mlx_conv2d(&y, x, w, 2, 2, 1, 1, 1, 1, 1, m->s));
    x = gelu(a, add(a, y, named(a, n, "bias")));
  }
  x = reshape(a, permute(a, x, (int[]){0, 2, 3, 1}, 4),
              (int[]){nc, ct, 480 * 16}, 3);
  x = linear(a, x, "thinker.audio_tower.conv_out", 0);
  mlx_array channels = fresh(a), times = fresh(a), inv = fresh(a), phase,
            sn = fresh(a), cs = fresh(a);
  TRY(a, mlx_arange(&channels, 0, m->enc_width / 2, 1, MLX_FLOAT32, m->s));
  TRY(a, mlx_exp(&inv, scale(a, channels, (float)(-log(10000.) / (m->enc_width / 2 - 1))), m->s));
  TRY(a, mlx_arange(&times, 0, ct, 1, MLX_FLOAT32, m->s));
  phase = mul(a, reshape(a, times, (int[]){ct, 1}, 2),
              reshape(a, inv, (int[]){1, m->enc_width / 2}, 2));
  TRY(a, mlx_sin(&sn, phase, m->s));
  TRY(a, mlx_cos(&cs, phase, m->s));
  x = add(a, x, join(a, (mlx_array[]){sn, cs}, 2, 1, 0));
  mlx_array *parts = host(a, (size_t)nc * sizeof(*parts));
  if (!parts) {
    a->why = "parts allocation";
    longjmp(a->jump, 1);
  }
  int nout = 0;
  for (int i = 0; i < nc; i++) {
    int len = f - i * 100;
    if (len > 100)
      len = 100;
    int n = (len + 7) / 8;
    nout += n;
    parts[i] =
        reshape(a, slice(a, x, (int[]){i, 0, 0}, (int[]){i + 1, n, m->enc_width}, 3),
                (int[]){n, m->enc_width}, 2);
  }
  x = join(a, parts, (size_t)nc, 0, 0);
  x = reshape(a, x, (int[]){1, nout, m->enc_width}, 3);
  /* The reference groups eight 100-frame convolution chunks per encoder window.
   */
  int window = ct * 8;
  float *mask = host(a, (size_t)nout * nout * sizeof(float));
  if (!mask) {
    a->why = "attention mask allocation";
    longjmp(a->jump, 1);
  }
  for (int i = 0; i < nout; i++)
    for (int j = 0; j < nout; j++)
      mask[(size_t)i * nout + j] = (i / window == j / window) ? 0.f : -1e9f;
  mlx_array am = data(a, mask, (int[]){1, 1, nout, nout}, 4, MLX_FLOAT32);
  for (int i = 0; i < m->enc_layers; i++)
    x = enc_layer(a, x, i, am);
  x = ln(a, x, "thinker.audio_tower.ln_post");
  x = gelu(a, linear(a, x, "thinker.audio_tower.proj1", 1));
  x = linear(a, x, "thinker.audio_tower.proj2", 1);
  x = reshape(a, x, (int[]){nout, m->hidden}, 2);
  TRY(a, mlx_array_eval(x));
  *out = mlx_array_new();
  TRY(a, mlx_array_set(out, x));
  clear(a);
  free(a);
  return 0;
}

static mlx_array decoder(arena *a, mlx_array x) {
  qwen_model_t *m = a->m;
  int t = mlx_array_dim(x, 1), offset = m->offset;
  for (int i = 0; i < 28; i++) {
    char p[120], n[180];
    snprintf(p, sizeof p, "thinker.model.layers.%d", i);
    snprintf(n, sizeof n, "%s.input_layernorm", p);
    mlx_array h = rms(a, x, n);
    snprintf(n, sizeof n, "%s.self_attn.q_proj", p);
    mlx_array q = reshape(a, linear(a, h, n, 0), (int[]){1, t, 16, 128}, 4);
    snprintf(n, sizeof n, "%s.self_attn.q_norm", p);
    q = rms(a, q, n);
    q = rope(a, permute(a, q, (int[]){0, 2, 1, 3}, 4), offset);
    snprintf(n, sizeof n, "%s.self_attn.k_proj", p);
    mlx_array k = reshape(a, linear(a, h, n, 0), (int[]){1, t, 8, 128}, 4);
    snprintf(n, sizeof n, "%s.self_attn.k_norm", p);
    k = rms(a, k, n);
    k = rope(a, permute(a, k, (int[]){0, 2, 1, 3}, 4), offset);
    snprintf(n, sizeof n, "%s.self_attn.v_proj", p);
    mlx_array v = heads(a, linear(a, h, n, 0), t, 8, 128);
    if (offset) {
      k = join(a, (mlx_array[]){m->k[i], k}, 2, 2, 0);
      v = join(a, (mlx_array[]){m->v[i], v}, 2, 2, 0);
    }
    if (!m->k[i].ctx)
      m->k[i] = mlx_array_new();
    if (!m->v[i].ctx)
      m->v[i] = mlx_array_new();
    TRY(a, mlx_array_set(&m->k[i], k));
    TRY(a, mlx_array_set(&m->v[i], v));
    h = unheads(a,
                sdpa(a, q, k, v, (float)(1. / sqrt(128.)),
                     t > 1 ? "causal" : "", (mlx_array){0}),
                t, 2048);
    snprintf(n, sizeof n, "%s.self_attn.o_proj", p);
    x = add(a, x, linear(a, h, n, 0));
    snprintf(n, sizeof n, "%s.post_attention_layernorm", p);
    h = rms(a, x, n);
    snprintf(n, sizeof n, "%s.mlp.gate_proj", p);
    mlx_array gate = silu(a, linear(a, h, n, 0));
    snprintf(n, sizeof n, "%s.mlp.up_proj", p);
    h = mul(a, gate, linear(a, h, n, 0));
    snprintf(n, sizeof n, "%s.mlp.down_proj", p);
    x = add(a, x, linear(a, h, n, 0));
  }
  x = rms(a, x, "thinker.model.norm");
  x = slice(a, x, (int[]){0, t - 1, 0}, (int[]){1, t, m->hidden}, 3);
  x = reshape(a, x, (int[]){1, m->hidden}, 2);
  mlx_array embed = weight(a, "thinker.model.embed_tokens.weight");
  x = mm(a, x, permute(a, embed, (int[]){1, 0}, 2));
  m->offset += t;
  return x;
}
int qwen_prefill(qwen_model_t *m, const int64_t *ids, size_t n,
                 qwen_array_t audio, qwen_array_t *out, char *err, size_t cap) {
  if (!m || !ids || !out || !audio.ctx || mlx_array_ndim(audio) != 2 || !n ||
      n > 4096)
    return -1;
  mlx_last_error[0] = 0;
  qwen_reset(m, NULL, 0);
  arena *a = calloc(1, sizeof(*a));
  if (!a)
    return -1;
  a->m = m;
  if (setjmp(a->jump)) {
    qwen_reset(m, NULL, 0);
    return failure(a, err, cap);
  }
  int audio_n = mlx_array_dim(audio, 0), count = 0;
  for (size_t i = 0; i < n; i++) {
    if (ids[i] < 0 || ids[i] >= 151936) {
      a->why = "prompt token out of range";
      longjmp(a->jump, 1);
    }
    count += ids[i] == 151676;
  }
  if (count != audio_n || mlx_array_dim(audio, 1) != m->hidden) {
    a->why = "audio placeholder count/shape mismatch";
    longjmp(a->jump, 1);
  }
  mlx_array embed = weight(a, "thinker.model.embed_tokens.weight"),
            ix = data(a, ids, (int[]){(int)n}, 1, MLX_INT64), tokens = fresh(a);
  TRY(a, mlx_take_axis(&tokens, embed, ix, 0, m->s));
  audio = cast(a, audio, mlx_array_dtype(tokens));
  mlx_array *parts = host(a, n * sizeof(*parts));
  if (!parts) {
    a->why = "prompt allocation";
    longjmp(a->jump, 1);
  }
  int j = 0;
  for (size_t i = 0; i < n; i++) {
    if (ids[i] == 151676) {
      parts[i] = slice(a, audio, (int[]){j, 0}, (int[]){j + 1, m->hidden}, 2);
      j++;
    } else
      parts[i] =
          slice(a, tokens, (int[]){(int)i, 0}, (int[]){(int)i + 1, m->hidden}, 2);
  }
  mlx_array x =
      reshape(a, join(a, parts, n, 0, 0), (int[]){1, (int)n, m->hidden}, 3);
  x = decoder(a, x);
  TRY(a, mlx_array_eval(x));
  *out = mlx_array_new();
  TRY(a, mlx_array_set(out, x));
  clear(a);
  free(a);
  return 0;
}
/* Internal fast path: token is a device argmax over this model's vocabulary.
 * Its UINT32 scalar/one-element shape is checked here. The argmax producer
 * guarantees values are in range; untrusted scalar IDs use qwen_step instead.
 * Output remains lazy; caller must check argmax/eval status and synchronize. */
int qwen_step_token_array(qwen_model_t *m, qwen_array_t token,
                          qwen_array_t *out, char *err, size_t cap) {
  if (!m || !out || !token.ctx || mlx_array_ndim(token) > 1 ||
      mlx_array_size(token) != 1 || mlx_array_dtype(token) != MLX_UINT32 ||
      m->offset <= 0 || m->offset >= 4096) {
    if (err && cap)
      snprintf(err, cap, "invalid device argmax token/state");
    return -1;
  }
  mlx_last_error[0] = 0;
  arena *a = calloc(1, sizeof(*a));
  if (!a)
    return -1;
  a->m = m;
  if (setjmp(a->jump)) {
    qwen_reset(m, NULL, 0);
    return failure(a, err, cap);
  }
  mlx_array x = fresh(a);
  TRY(a, mlx_take_axis(&x, weight(a, "thinker.model.embed_tokens.weight"),
                       token, 0, m->s));
  x = reshape(a, x, (int[]){1, 1, m->hidden}, 3);
  x = decoder(a, x);
  *out = mlx_array_new();
  TRY(a, mlx_array_set(out, x));
  clear(a);
  free(a);
  return 0;
}
int qwen_step(qwen_model_t *m, uint32_t token, qwen_array_t *out, char *err,
              size_t cap) {
  if (token >= 151936) {
    if (err && cap)
      snprintf(err, cap, "token ID out of range");
    return -1;
  }
  mlx_array ix = mlx_array_new_data(&token, (int[]){1}, 1, MLX_UINT32);
  if (!ix.ctx) {
    if (err && cap)
      snprintf(err, cap, "token allocation failed");
    return -1;
  }
  int rc = qwen_step_token_array(m, ix, out, err, cap);
  mlx_array_free(ix);
  return rc;
}
