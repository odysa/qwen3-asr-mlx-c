/* MIT License

Copyright (c) 2026 Salvatore Sanfilippo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
/*
 * qwen_asr_tokenizer.h - Qwen2/Qwen3 BPE tokenizer (GPT-2 byte-level)
 *
 * Supports:
 *  - decode token ID -> UTF-8 text
 *  - encode UTF-8 text -> token IDs (vocab.json + merges.txt)
 */

#ifndef QWEN_ASR_TOKENIZER_H
#define QWEN_ASR_TOKENIZER_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    char **id_to_text;   /* [vocab_size] decoded text strings */
    char **id_to_bpe;    /* [vocab_size] raw BPE token strings from vocab.json */
    int vocab_size;

    /* Internal hash maps (opaque to callers) */
    void *vocab_map;
    int vocab_map_cap;
    void *merge_map;
    int merge_map_cap;
} qwen_tokenizer_t;

/* Load tokenizer from vocab.json in model directory */
qwen_tokenizer_t *qwen_tokenizer_load(const char *vocab_json_path);

/* Decode a single token ID to text. Returns pointer to internal string. */
const char *qwen_tokenizer_decode(const qwen_tokenizer_t *tok, int token_id);

/* Encode one pretokenized UTF-8 span into token IDs using BPE.
 * Does not implement general Qwen regex pretokenization/system/hotwords.
 * Use qwen_tokenizer_prompt for the supported model-specific ASR prompts.
 * Returns malloc'd array of token IDs and sets *out_n_tokens.
 * Returns NULL on error (and sets *out_n_tokens to 0). */
int *qwen_tokenizer_encode(const qwen_tokenizer_t *tok, const char *text, int *out_n_tokens);

/* Free tokenizer */
void qwen_tokenizer_free(qwen_tokenizer_t *tok);

/* Model-specific ChatML audio prompt; language NULL means auto-detect. */
int64_t *qwen_tokenizer_prompt(const qwen_tokenizer_t *, size_t audio_tokens, const char *language, size_t *count);
/* Concatenate output token bytes (UTF-8 may span tokens); caller frees. */
/* Added token literals are retained; caller handles EOS and auto-language prefix. */
char *qwen_tokenizer_decode_all(const qwen_tokenizer_t *, const int64_t *, size_t count);
#endif /* QWEN_ASR_TOKENIZER_H */
