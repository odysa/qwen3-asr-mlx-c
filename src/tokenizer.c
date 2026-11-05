/* Adapted from antirez/qwen-asr 924694251d9e0f18e5d86bbd06aa3ab5f870002d
MIT License

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
 * qwen_asr_tokenizer.c - Qwen2/Qwen3 BPE tokenizer (GPT-2 byte-level)
 *
 * Supports:
 *   - decode token IDs to UTF-8 text (for ASR output assembly)
 *   - encode UTF-8 text to token IDs using vocab.json + merges.txt
 */

#include "tokenizer.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * GPT-2 Bytes-to-Unicode Mapping
 * ======================================================================== */

static int gpt2_byte_to_unicode[256];
static int gpt2_unicode_to_byte[512]; /* codepoints up to ~384 */
static int gpt2_mapping_initialized = 0;

static void init_gpt2_mapping(void) {
    if (gpt2_mapping_initialized) return;

    memset(gpt2_unicode_to_byte, -1, sizeof(gpt2_unicode_to_byte));

    int n = 0;
    for (int b = 0; b < 256; b++) {
        int is_normal = 0;
        if (b >= 33 && b <= 126) is_normal = 1;   /* '!'..'~' */
        if (b >= 161 && b <= 172) is_normal = 1;  /* '¡'..'¬' */
        if (b >= 174 && b <= 255) is_normal = 1;  /* '®'..'ÿ' */

        if (is_normal) gpt2_byte_to_unicode[b] = b;
        else gpt2_byte_to_unicode[b] = 256 + n++;
    }

    for (int b = 0; b < 256; b++) {
        int cp = gpt2_byte_to_unicode[b];
        if (cp < 512) gpt2_unicode_to_byte[cp] = b;
    }

    gpt2_mapping_initialized = 1;
}

/* Convert one Unicode codepoint to UTF-8 bytes. Returns bytes written. */
static int utf8_encode_cp(int cp, char out[4]) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

/*
 * Decode a GPT-2 encoded token string (vocab key) to raw bytes/UTF-8 text.
 * Returns allocated string (caller must free).
 */
static char *decode_gpt2_token(const char *token_str) {
    init_gpt2_mapping();

    size_t len = strlen(token_str);
    unsigned char *bytes = (unsigned char *)malloc(len + 1);
    if (!bytes) return NULL;

    int byte_count = 0;
    const unsigned char *p = (const unsigned char *)token_str;
    const unsigned char *end = p + len;

    while (p < end) {
        int cp;
        int nbytes;

        if ((*p & 0x80) == 0) {
            cp = *p;
            nbytes = 1;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6;
            if (p + 1 < end) cp |= (p[1] & 0x3F);
            nbytes = 2;
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12;
            if (p + 1 < end) cp |= (p[1] & 0x3F) << 6;
            if (p + 2 < end) cp |= (p[2] & 0x3F);
            nbytes = 3;
        } else if ((*p & 0xF8) == 0xF0) {
            cp = (*p & 0x07) << 18;
            if (p + 1 < end) cp |= (p[1] & 0x3F) << 12;
            if (p + 2 < end) cp |= (p[2] & 0x3F) << 6;
            if (p + 3 < end) cp |= (p[3] & 0x3F);
            nbytes = 4;
        } else {
            cp = *p;
            nbytes = 1;
        }
        p += nbytes;

        if (cp < 512 && gpt2_unicode_to_byte[cp] >= 0) {
            bytes[byte_count++] = (unsigned char)gpt2_unicode_to_byte[cp];
        } else {
            bytes[byte_count++] = '?';
        }
    }
    bytes[byte_count] = '\0';
    return (char *)bytes;
}

/* ========================================================================
 * Simple JSON parser for vocab.json
 * ======================================================================== */

static void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

static int hexval(unsigned char c){
 if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;if(c>='A'&&c<='F')return c-'A'+10;return -1;
}
static int parse_json_string(const char **p,char *out,size_t cap,const char *limit){
 skip_ws(p);const unsigned char *q=(const unsigned char *)*p;
 const unsigned char *end=(const unsigned char *)limit;size_t n=0;
 if(q>end)return -1;
 if(q==end||*q++!='"'||cap==0)return -1;
 while(q<end){
  unsigned char c=*q++;if(c=='"'){out[n]=0;*p=(const char *)q;return 0;}
  if(c<32)return -1;
  if(c!='\\'){if(n+1>=cap)return -1;out[n++]=(char)c;continue;}
  if(q==end)return -1;c=*q++;
  if(c=='u'){
   if(end-q<4)return -1;unsigned cp=0;
   for(int j=0;j<4;j++){int h=hexval(q[j]);if(h<0)return -1;cp=(cp<<4)|(unsigned)h;}q+=4;
   if(cp>=0xd800&&cp<=0xdbff){
    if(end-q<6||q[0]!='\\'||q[1]!='u')return -1;q+=2;unsigned low=0;
    for(int j=0;j<4;j++){int h=hexval(q[j]);if(h<0)return -1;low=(low<<4)|(unsigned)h;}q+=4;
    if(low<0xdc00||low>0xdfff)return -1;cp=0x10000+((cp-0xd800)<<10)+(low-0xdc00);
   }else if(cp>=0xdc00&&cp<=0xdfff)return -1;
   /* Embedded NUL cannot be represented by this C-string vocabulary API. */
   if(cp==0)return -1;
   unsigned char bytes[4];size_t k;
   if(cp<0x80){bytes[0]=(unsigned char)cp;k=1;}
   else if(cp<0x800){bytes[0]=0xc0|(cp>>6);bytes[1]=0x80|(cp&63);k=2;}
   else if(cp<0x10000){bytes[0]=0xe0|(cp>>12);bytes[1]=0x80|((cp>>6)&63);bytes[2]=0x80|(cp&63);k=3;}
   else{bytes[0]=0xf0|(cp>>18);bytes[1]=0x80|((cp>>12)&63);bytes[2]=0x80|((cp>>6)&63);bytes[3]=0x80|(cp&63);k=4;}
   if(k>=cap-n)return -1;memcpy(out+n,bytes,k);n+=k;continue;
  }
  switch(c){case 'n':c='\n';break;case 't':c='\t';break;case 'r':c='\r';break;case 'b':c='\b';break;case 'f':c='\f';break;case '"':case '\\':case '/':break;default:return -1;}
  if(n+1>=cap)return -1;out[n++]=(char)c;
 }
 return -1;
}

static int64_t parse_json_int(const char **p) {
    skip_ws(p);
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    if(**p<'0'||**p>'9')return INT64_MIN;
    int64_t val = 0;
    while (**p >= '0' && **p <= '9') {
        if(val>(INT64_MAX-(**p-'0'))/10)return INT64_MIN;
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

/* ========================================================================
 * String->int hash map (open addressing)
 * ======================================================================== */

typedef struct {
    char *key; /* NULL means empty slot */
    int value;
} str_int_entry_t;

static uint64_t fnv1a_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

static int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

static int map_insert(str_int_entry_t *map, int cap, char *key, int value) {
    if (!map || cap <= 0 || !key) return -1;
    int mask = cap - 1;
    int pos = (int)(fnv1a_hash(key) & (uint64_t)mask);
    for (int i = 0; i < cap; i++) {
        int idx = (pos + i) & mask;
        if (!map[idx].key) {
            map[idx].key = key;
            map[idx].value = value;
            return 0;
        }
        if (strcmp(map[idx].key, key) == 0) {
            map[idx].value = value;
            return 0;
        }
    }
    return -1;
}

static int map_get(const str_int_entry_t *map, int cap, const char *key) {
    if (!map || cap <= 0 || !key) return -1;
    int mask = cap - 1;
    int pos = (int)(fnv1a_hash(key) & (uint64_t)mask);
    for (int i = 0; i < cap; i++) {
        int idx = (pos + i) & mask;
        if (!map[idx].key) return -1;
        if (strcmp(map[idx].key, key) == 0) return map[idx].value;
    }
    return -1;
}

/* ========================================================================
 * BPE helpers
 * ======================================================================== */

static int utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static int split_utf8_symbols(const char *s, char ***out_syms, int *out_n) {
    *out_syms = NULL;
    *out_n = 0;
    if (!s || !*s) return 0;

    int n = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ) {
        int l = utf8_char_len(*p);
        p += l;
        n++;
    }
    if (n <= 0) return 0;

    char **syms = (char **)calloc((size_t)n, sizeof(char *));
    if (!syms) return -1;

    int i = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ) {
        int l = utf8_char_len(*p);
        syms[i] = (char *)malloc((size_t)l + 1);
        if (!syms[i]) {
            for (int j = 0; j < i; j++) free(syms[j]);
            free(syms);
            return -1;
        }
        memcpy(syms[i], p, (size_t)l);
        syms[i][l] = '\0';
        p += l;
        i++;
    }

    *out_syms = syms;
    *out_n = n;
    return 0;
}

static char *str_concat2(const char *a, const char *b) {
    size_t la = strlen(a), lb = strlen(b);
    char *r = (char *)malloc(la + lb + 1);
    if (!r) return NULL;
    memcpy(r, a, la);
    memcpy(r + la, b, lb);
    r[la + lb] = '\0';
    return r;
}

static int merge_rank(const qwen_tokenizer_t *tok, const char *a, const char *b) {
    if (!tok->merge_map || tok->merge_map_cap <= 0) return INT_MAX;

    size_t la = strlen(a), lb = strlen(b);
    char *pair = (char *)malloc(la + 1 + lb + 1);
    if (!pair) return INT_MAX;
    memcpy(pair, a, la);
    pair[la] = ' ';
    memcpy(pair + la + 1, b, lb);
    pair[la + 1 + lb] = '\0';

    int rank = map_get((const str_int_entry_t *)tok->merge_map, tok->merge_map_cap, pair);
    free(pair);
    return rank >= 0 ? rank : INT_MAX;
}

static int append_id(int **arr, int *n, int *cap, int id) {
    if (*n >= *cap) {
        int new_cap = (*cap == 0) ? 16 : (*cap * 2);
        int *tmp = (int *)realloc(*arr, (size_t)new_cap * sizeof(int));
        if (!tmp) return -1;
        *arr = tmp;
        *cap = new_cap;
    }
    (*arr)[(*n)++] = id;
    return 0;
}

/* Convert UTF-8 bytes to GPT-2 byte-level unicode string. */
static char *text_to_bpe_unicode(const char *text) {
    init_gpt2_mapping();
    size_t len = strlen(text);
    /* Every input byte becomes one codepoint up to 3 UTF-8 bytes. */
    char *out = (char *)malloc(len * 3 + 1);
    if (!out) return NULL;

    size_t w = 0;
    for (size_t i = 0; i < len; i++) {
        int cp = gpt2_byte_to_unicode[(unsigned char)text[i]];
        char tmp[4];
        int n = utf8_encode_cp(cp, tmp);
        for (int j = 0; j < n; j++) out[w++] = tmp[j];
    }
    out[w] = '\0';
    return out;
}

/* Encode one mapped BPE unicode string to token IDs. */
static int encode_bpe_word(const qwen_tokenizer_t *tok, const char *mapped, int **out_ids, int *out_n) {
    *out_ids = NULL;
    *out_n = 0;
    if (!mapped || !*mapped) return 0;

    char **syms = NULL;
    int n_syms = 0;
    if (split_utf8_symbols(mapped, &syms, &n_syms) != 0) return -1;
    if (n_syms <= 0) {
        free(syms);
        return 0;
    }

    while (n_syms > 1) {
        int best_rank = INT_MAX;
        int best_i = -1;
        for (int i = 0; i < n_syms - 1; i++) {
            int r = merge_rank(tok, syms[i], syms[i + 1]);
            if (r < best_rank) {
                best_rank = r;
                best_i = i;
            }
        }
        if (best_i < 0 || best_rank == INT_MAX) break;

        char *merged = str_concat2(syms[best_i], syms[best_i + 1]);
        if (!merged) {
            for (int i = 0; i < n_syms; i++) free(syms[i]);
            free(syms);
            return -1;
        }
        free(syms[best_i]);
        free(syms[best_i + 1]);
        syms[best_i] = merged;
        for (int j = best_i + 1; j < n_syms - 1; j++) syms[j] = syms[j + 1];
        n_syms--;
    }

    int *ids = NULL;
    int n_ids = 0, cap = 0;
    for (int i = 0; i < n_syms; i++) {
        int id = map_get((const str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap, syms[i]);
        if (id < 0) {
            /* Should not happen with valid vocab + merges + byte-level mapping. */
            for (int k = 0; k < n_syms; k++) free(syms[k]);
            free(syms);
            free(ids);
            return -1;
        }
        if (append_id(&ids, &n_ids, &cap, id) != 0) {
            for (int k = 0; k < n_syms; k++) free(syms[k]);
            free(syms);
            free(ids);
            return -1;
        }
    }

    for (int i = 0; i < n_syms; i++) free(syms[i]);
    free(syms);

    *out_ids = ids;
    *out_n = n_ids;
    return 0;
}

static int derive_merges_path(const char *vocab_path, char *out_path, size_t out_cap) {
    const char *slash = strrchr(vocab_path, '/');
    if (!slash) {
        if (snprintf(out_path, out_cap, "merges.txt") >= (int)out_cap) return -1;
        return 0;
    }
    size_t dir_len = (size_t)(slash - vocab_path);
    if (dir_len + 12 >= out_cap) return -1;
    memcpy(out_path, vocab_path, dir_len);
    out_path[dir_len] = '\0';
    snprintf(out_path + dir_len, out_cap - dir_len, "/merges.txt");
    return 0;
}

static void trim_newline(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
}

static int parse_merge_pair(char *line, char **a, char **b) {
    *a = NULL;
    *b = NULL;
    trim_newline(line);
    if (line[0] == '\0') return -1;
    if (line[0] == '#') return -1;

    char *sp = strchr(line, ' ');
    if (!sp) return -1;
    *sp = '\0';
    char *p2 = sp + 1;
    while (*p2 == ' ') p2++;
    if (*line == '\0' || *p2 == '\0') return -1;
    *a = line;
    *b = p2;
    return 0;
}

static int load_merges_map(qwen_tokenizer_t *tok, const char *merges_path) {
    FILE *f = fopen(merges_path, "rb");
    if (!f) return -1;

    int n_pairs = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        char *a = NULL, *b = NULL;
        if (parse_merge_pair(line, &a, &b) == 0) n_pairs++;
    }
    if (n_pairs <= 0) {
        fclose(f);
        return -1;
    }

    tok->merge_map_cap = next_pow2(n_pairs * 2);
    tok->merge_map = calloc((size_t)tok->merge_map_cap, sizeof(str_int_entry_t));
    if (!tok->merge_map) {
        fclose(f);
        tok->merge_map_cap = 0;
        return -1;
    }

    rewind(f);
    int rank = 0;
    while (fgets(line, sizeof(line), f)) {
        char *a = NULL, *b = NULL;
        if (parse_merge_pair(line, &a, &b) != 0) continue;

        size_t la = strlen(a), lb = strlen(b);
        char *key = (char *)malloc(la + 1 + lb + 1);
        if (!key) continue;
        memcpy(key, a, la);
        key[la] = ' ';
        memcpy(key + la + 1, b, lb);
        key[la + 1 + lb] = '\0';

        if (map_insert((str_int_entry_t *)tok->merge_map, tok->merge_map_cap, key, rank) != 0) {
            free(key);
        }
        rank++;
    }
    fclose(f);
    return 0;
}

/* ========================================================================
 * Public API
 * ======================================================================== */

qwen_tokenizer_t *qwen_tokenizer_load(const char *vocab_json_path) {
    FILE *f = fopen(vocab_json_path, "rb");
    if (!f) {
        fprintf(stderr, "qwen_tokenizer_load: cannot open %s\n", vocab_json_path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    if(file_size<2||file_size>100000000){fclose(f);return NULL;}
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc((size_t)file_size + 1);
    if (!json || fread(json, 1, (size_t)file_size, f) != (size_t)file_size) {
        fclose(f);
        free(json);
        return NULL;
    }
    fclose(f);
    json[file_size] = '\0';

    int max_id = 0;
    const char *p = json;
    skip_ws(&p);
    if (*p != '{') {
        free(json);
        return NULL;
    }
    p++;

    const char *saved = p;
    skip_ws(&p);if(*p=='}'){free(json);return NULL;}
    for(;;){
        char key[4096];
        if(parse_json_string(&p,key,sizeof(key),json+file_size)){free(json);return NULL;}
        skip_ws(&p);if(*p!=':'){free(json);return NULL;}p++;
        int64_t id=parse_json_int(&p);
        if(id<0||id>2000000){free(json);return NULL;}
        if(id>max_id)max_id=(int)id;
        skip_ws(&p);if(*p=='}'){p++;break;}
        if(*p!=','){free(json);return NULL;}p++;skip_ws(&p);
        if(!*p||*p=='}'){free(json);return NULL;}
    }
    skip_ws(&p);if(*p){free(json);return NULL;}
    int vocab_size = max_id + 1;
    qwen_tokenizer_t *tok = (qwen_tokenizer_t *)calloc(1, sizeof(qwen_tokenizer_t));
    if (!tok) {
        free(json);
        return NULL;
    }
    tok->vocab_size = vocab_size;
    tok->id_to_text = (char **)calloc((size_t)vocab_size, sizeof(char *));
    tok->id_to_bpe = (char **)calloc((size_t)vocab_size, sizeof(char *));
    if (!tok->id_to_text || !tok->id_to_bpe) {
        qwen_tokenizer_free(tok);
        free(json);
        return NULL;
    }

    p = saved;
    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char key[4096];
        if (parse_json_string(&p, key, sizeof(key), json+file_size) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        int64_t id = parse_json_int(&p);

        if (id >= 0 && id < vocab_size) {
            free(tok->id_to_bpe[id]);
            free(tok->id_to_text[id]);
            tok->id_to_bpe[id] = strdup(key);
            tok->id_to_text[id] = decode_gpt2_token(key);
        }
    }
    free(json);

    int n_vocab_entries = 0;
    for (int i = 0; i < vocab_size; i++) if (tok->id_to_bpe[i]) n_vocab_entries++;
    if(n_vocab_entries==0){qwen_tokenizer_free(tok);return NULL;}
    tok->vocab_map_cap = next_pow2(n_vocab_entries * 2 + 1);
    tok->vocab_map = calloc((size_t)tok->vocab_map_cap, sizeof(str_int_entry_t));
    if (!tok->vocab_map) {
        qwen_tokenizer_free(tok);
        return NULL;
    }
    for (int i = 0; i < vocab_size; i++) {
        if (tok->id_to_bpe[i]) {
            map_insert((str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap, tok->id_to_bpe[i], i);
        }
    }

    char merges_path[1024];
    if (derive_merges_path(vocab_json_path, merges_path, sizeof(merges_path)) != 0) {qwen_tokenizer_free(tok);return NULL;}
    {
        if (load_merges_map(tok, merges_path) != 0) {
            qwen_tokenizer_free(tok); return NULL;
        }
    }

    return tok;
}

const char *qwen_tokenizer_decode(const qwen_tokenizer_t *tok, int token_id) {
    if(tok){switch(token_id){
    case 151643: return "<|endoftext|>";
    case 151644: return "<|im_start|>";
    case 151645: return "<|im_end|>";
    case 151646: return "<|object_ref_start|>";
    case 151647: return "<|object_ref_end|>";
    case 151648: return "<|box_start|>";
    case 151649: return "<|box_end|>";
    case 151650: return "<|quad_start|>";
    case 151651: return "<|quad_end|>";
    case 151652: return "<|vision_start|>";
    case 151653: return "<|vision_end|>";
    case 151654: return "<|vision_pad|>";
    case 151655: return "<|image_pad|>";
    case 151656: return "<|video_pad|>";
    case 151657: return "<tool_call>";
    case 151658: return "</tool_call>";
    case 151659: return "<|fim_prefix|>";
    case 151660: return "<|fim_middle|>";
    case 151661: return "<|fim_suffix|>";
    case 151662: return "<|fim_pad|>";
    case 151663: return "<|repo_name|>";
    case 151664: return "<|file_sep|>";
    case 151665: return "<tool_response>";
    case 151666: return "</tool_response>";
    case 151667: return "<think>";
    case 151668: return "</think>";
    case 151669: return "<|audio_start|>";
    case 151670: return "<|audio_end|>";
    case 151671: return "<tts_pad>";
    case 151672: return "<tts_text_bos>";
    case 151673: return "<tts_text_eod>";
    case 151674: return "<tts_text_bos_single>";
    case 151675: return "<non_speech>";
    case 151676: return "<|audio_pad|>";
    case 151677: return "<blank1>";
    case 151678: return "<blank2>";
    case 151679: return "<blank3>";
    case 151680: return "<blank4>";
    case 151681: return "<blank5>";
    case 151682: return "<blank6>";
    case 151683: return "<blank7>";
    case 151684: return "<blank8>";
    case 151685: return "<blank9>";
    case 151686: return "<blank10>";
    case 151687: return "<blank11>";
    case 151688: return "<blank12>";
    case 151689: return "<blank13>";
    case 151690: return "<blank14>";
    case 151691: return "<blank15>";
    case 151692: return "<blank16>";
    case 151693: return "<blank17>";
    case 151694: return "<blank18>";
    case 151695: return "<blank19>";
    case 151696: return "<blank20>";
    case 151697: return "<blank21>";
    case 151698: return "<blank22>";
    case 151699: return "<blank23>";
    case 151700: return "<blank24>";
    case 151701: return "<blank25>";
    case 151702: return "<blank26>";
    case 151703: return "<blank27>";
    case 151704: return "<asr_text>";
    default:break;}}
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) return "";
    return tok->id_to_text[token_id] ? tok->id_to_text[token_id] : "";
}

int *qwen_tokenizer_encode(const qwen_tokenizer_t *tok, const char *text, int *out_n_tokens) {
    if (out_n_tokens) *out_n_tokens = 0;
    if (!tok || !text || text[0] == '\0') return NULL;

    char *mapped = text_to_bpe_unicode(text);
    if (!mapped) return NULL;

    int *ids = NULL;
    int n_ids = 0;
    if (encode_bpe_word(tok, mapped, &ids, &n_ids) != 0) {
        free(mapped);
        free(ids);
        return NULL;
    }
    free(mapped);

    if (out_n_tokens) *out_n_tokens = n_ids;
    return ids;
}

void qwen_tokenizer_free(qwen_tokenizer_t *tok) {
    if (!tok) return;

    if (tok->id_to_text) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->id_to_text[i]);
        free(tok->id_to_text);
    }
    if (tok->id_to_bpe) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->id_to_bpe[i]);
        free(tok->id_to_bpe);
    }

    if (tok->merge_map) {
        str_int_entry_t *m = (str_int_entry_t *)tok->merge_map;
        for (int i = 0; i < tok->merge_map_cap; i++) free(m[i].key);
        free(tok->merge_map);
    }
    free(tok->vocab_map);
    free(tok);
}

/* Official Qwen3-ASR added-token IDs, pinned tokenizer_config.json. */
static int prompt_push(int64_t **ids,size_t *n,size_t *cap,int64_t id){
 if(*n==*cap){size_t c=*cap?*cap*2:64;if(c>10000000)return -1;int64_t *v=realloc(*ids,c*sizeof(*v));if(!v)return -1;*ids=v;*cap=c;}(*ids)[(*n)++]=id;return 0;
}
static int prompt_text(const qwen_tokenizer_t *t,int64_t **ids,size_t *n,size_t *cap,const char *text){
 int count=0;int *v=qwen_tokenizer_encode(t,text,&count);if(!v)return -1;
 for(int i=0;i<count;i++)if(prompt_push(ids,n,cap,v[i])){free(v);return -1;}free(v);return 0;
}
int64_t *qwen_tokenizer_prompt(const qwen_tokenizer_t *t,size_t audio_tokens,const char *language,size_t *count){
 if(count)*count=0;if(!t||!count||audio_tokens==0||audio_tokens>1000000)return NULL;
 /* Initial CLI scope: forced English/Chinese or NULL auto-detection. */
 if(language&&strcmp(language,"English")&&strcmp(language,"Chinese"))return NULL;
 int64_t *v=NULL;size_t n=0,cap=0;
 #define PUSH(x) do{if(prompt_push(&v,&n,&cap,(x)))goto fail;}while(0)
 #define TEXT(x) do{if(prompt_text(t,&v,&n,&cap,(x)))goto fail;}while(0)
 PUSH(151644);TEXT("system\n");PUSH(151645);TEXT("\n");
 PUSH(151644);TEXT("user\n");PUSH(151669);
 for(size_t i=0;i<audio_tokens;i++)PUSH(151676);
 PUSH(151670);PUSH(151645);TEXT("\n");PUSH(151644);TEXT("assistant\n");
 if(language){TEXT("language");char b[32];snprintf(b,sizeof(b)," %s",language);TEXT(b);PUSH(151704);}
 *count=n;return v;
 fail:free(v);return NULL;
 #undef PUSH
 #undef TEXT
}
char *qwen_tokenizer_decode_all(const qwen_tokenizer_t *t,const int64_t *ids,size_t count){
 if(!t||(!ids&&count))return NULL;size_t total=0;
 for(size_t i=0;i<count;i++){if(ids[i]<0||ids[i]>INT_MAX)return NULL;const char *s=qwen_tokenizer_decode(t,(int)ids[i]);size_t k=strlen(s);if(!k && ids[i]>=t->vocab_size)return NULL;if(k>SIZE_MAX-total-1)return NULL;total+=k;}
 char *v=malloc(total+1);if(!v)return NULL;size_t off=0;
 for(size_t i=0;i<count;i++){const char *s=qwen_tokenizer_decode(t,(int)ids[i]);size_t k=strlen(s);memcpy(v+off,s,k);off+=k;}v[off]=0;return v;
}
