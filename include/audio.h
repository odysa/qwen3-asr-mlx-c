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
#ifndef QWEN_AUDIO_H
#define QWEN_AUDIO_H
#include <stddef.h>
#include <stdint.h>
/* Only16kHz mono PCM16 accepted. Caller frees buffers; failure returns NULL. */
float *qwen_load_wav(const char *,int *);
float *qwen_parse_wav_buffer(const uint8_t *,size_t,int *);
/* 128xF row-major F=floor(samples/160), float32 log-mel. */
float *qwen_mel_spectrogram(const float *,int,int *);
#endif
