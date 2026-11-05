#ifndef QWEN_WEIGHTS_H
#define QWEN_WEIGHTS_H
#include <stddef.h>
#include <stdint.h>
#ifdef QWEN_WITH_MLX_C
#include "mlx/c/io.h"
#endif

typedef enum { QWEN_DTYPE_UNKNOWN=0, QWEN_DTYPE_BF16=1, QWEN_DTYPE_F16=2, QWEN_DTYPE_F32=3 } qwen_dtype_t;
typedef struct { char *name; uint64_t offset; uint64_t nbytes; uint32_t ndim; uint64_t *shape; qwen_dtype_t dtype; } qwen_tensor_t;
typedef struct { void *data; size_t size; qwen_tensor_t *tensors; size_t ntensors; } qwen_weights_t;
int qwen_weights_load(const char *path, qwen_weights_t **out, char *err, size_t errcap);
const qwen_tensor_t *qwen_weights_find(const qwen_weights_t *, const char *name);
void qwen_weights_free(qwen_weights_t *);
#ifdef QWEN_WITH_MLX_C
/* Loads original safetensors through MLX C; caller owns the returned maps. */
int qwen_weights_load_mlx(const char *path, mlx_map_string_to_array *tensors,
                          mlx_map_string_to_string *metadata, char *err, size_t errcap);
#endif
#endif
