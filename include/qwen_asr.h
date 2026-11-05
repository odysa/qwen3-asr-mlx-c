#ifndef QWEN_ASR_H
#define QWEN_ASR_H
#include <stddef.h>
#include <stdint.h>
#include "weights.h"

typedef struct qwen_model qwen_model_t;
typedef struct { unsigned threads; } qwen_model_opts_t;
#ifdef QWEN_WITH_MLX_C
typedef mlx_array qwen_array_t;
#else
typedef struct { void *ctx; } qwen_array_t;
#endif
typedef struct { const uint32_t *ids; size_t count; int finished; } qwen_tokens_t;
int qwen_model_load(const char *path, const qwen_model_opts_t *, qwen_model_t **out, char *err, size_t errcap);
int qwen_encode(qwen_model_t *, qwen_array_t mel_128xf, qwen_array_t *audio_embeddings, char *err, size_t errcap);
int qwen_prefill(qwen_model_t *, const int64_t *prompt_ids, size_t prompt_len, qwen_array_t audio_embeddings, qwen_array_t *last_logits, char *err, size_t errcap);
/* Asynchronous callers must retain a lookahead slot for the returned logits
 * and synchronize the owning stream before freeing arrays or resetting state. */
int qwen_step_token_array(qwen_model_t *, qwen_array_t token, qwen_array_t *next_logits, char *err, size_t errcap);
int qwen_step(qwen_model_t *, uint32_t token_id, qwen_array_t *next_logits, char *err, size_t errcap);
int qwen_reset(qwen_model_t *, char *err, size_t errcap);
void qwen_model_free(qwen_model_t *);
#endif
