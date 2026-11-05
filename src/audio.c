/* Slaney filter and STFT adapted from antirez/qwen-asr 924694251d9e0f18e5d86bbd06aa3ab5f870002d
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
#include "audio.h"
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#define N_MEL 128
#define N_FREQ 201
#define SAMPLE_RATE 16000
#define N_FFT 400
#define HOP_LENGTH 160
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static uint16_t u16(const uint8_t *p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
static uint32_t u32(const uint8_t *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
float *qwen_parse_wav_buffer(const uint8_t *p,size_t n,int *count){
 if(count)*count=0;
 if(!count||!p||n<12||memcmp(p,"RIFF",4)||memcmp(p+8,"WAVE",4)||u32(p+4)!=(n-8))return NULL;
 const uint8_t *data=NULL;size_t bytes=0;int fmt=0;
 size_t i=12;
 for(;i+8<=n;){uint32_t k=u32(p+i+4);const uint8_t *b=p+i+8;if(k>n-i-8)return NULL;
  if(!memcmp(p+i,"fmt ",4)){if(fmt||k<16||u16(b)!=1||u16(b+2)!=1||u32(b+4)!=16000||u32(b+8)!=32000||u16(b+12)!=2||u16(b+14)!=16)return NULL;fmt=1;}
  if(!memcmp(p+i,"data",4)){if(data)return NULL;data=b;bytes=k;}
  size_t step=8+(size_t)k+(k&1);if(step>n-i)return NULL;i+=step;
 }
 if(i!=n||!fmt||!data||bytes<320||(bytes&1)||bytes/2>19200000)return NULL;
 float *x=malloc(bytes/2*sizeof(float));if(!x)return NULL;
 for(size_t j=0;j<bytes/2;j++)x[j]=(float)(int16_t)u16(data+2*j)/32768.0f;
 *count=(int)(bytes/2);return x;
}
float *qwen_load_wav(const char *path,int *count){
 if(count)*count=0;if(!path||!count)return NULL;FILE *f=fopen(path,"rb");if(!f)return NULL;
 if(fseek(f,0,SEEK_END)){fclose(f);return NULL;}long len=ftell(f);if(len<12||len>40000000||fseek(f,0,SEEK_SET)){fclose(f);return NULL;}
 uint8_t *b=malloc((size_t)len);if(!b){fclose(f);return NULL;}if(fread(b,1,(size_t)len,f)!=(size_t)len){free(b);fclose(f);return NULL;}fclose(f);
 float *x=qwen_parse_wav_buffer(b,(size_t)len,count);free(b);return x;
}

static double hertz_to_mel(double freq) {
    const double min_log_hertz = 1000.0f;
    const double min_log_mel = 15.0f;
    const double logstep = 27.0f / log(6.4f);
    double mels = 3.0f * freq / 200.0f;
    if (freq >= min_log_hertz) mels = min_log_mel + log(freq / min_log_hertz) * logstep;
    return mels;
}

static double mel_to_hertz(double mels) {
    const double min_log_hertz = 1000.0f;
    const double min_log_mel = 15.0f;
    const double logstep = log(6.4f) / 27.0f;
    double freq = 200.0f * mels / 3.0f;
    if (mels >= min_log_mel) freq = min_log_hertz * exp(logstep * (mels - min_log_mel));
    return freq;
}

static double *build_mel_filters(void) {
    double *filters = (double *)calloc((size_t)N_MEL * N_FREQ, sizeof(double));
    if (!filters) return NULL;

    double fft_freqs[N_FREQ];
    for (int i = 0; i < N_FREQ; i++)
        fft_freqs[i] = (double)i * ((double)SAMPLE_RATE / 2.0f) / (double)(N_FREQ - 1);

    double mel_min = hertz_to_mel(0.0f);
    double mel_max = hertz_to_mel((double)SAMPLE_RATE / 2.0f);

    double filter_freqs[N_MEL + 2];
    double filter_diff[N_MEL + 1];
    for (int i = 0; i < N_MEL + 2; i++) {
        double mel = mel_min + (mel_max - mel_min) * (double)i / (double)(N_MEL + 1);
        filter_freqs[i] = mel_to_hertz(mel);
    }
    for (int i = 0; i < N_MEL + 1; i++) {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
        if (filter_diff[i] == 0.0f) filter_diff[i] = 1e-6f;
    }

    for (int m = 0; m < N_MEL; m++) {
        double enorm = 2.0f / (filter_freqs[m + 2] - filter_freqs[m]);
        for (int f = 0; f < N_FREQ; f++) {
            double down = (fft_freqs[f] - filter_freqs[m]) / filter_diff[m];
            double up = (filter_freqs[m + 2] - fft_freqs[f]) / filter_diff[m + 1];
            double val = fmin(down, up);
            if (val < 0.0f) val = 0.0f;
            filters[(size_t)m * N_FREQ + f] = val * enorm;
        }
    }
    return filters;
}


/* A 400-point DFT is five stride-5 80-point DFTs plus a radix-5
 * twiddle combine. Runtime Accelerate accepts 80 but rejects 400 on this Mac;
 * no transform-size substitution or zero padding is involved. */
static void dft400_power(const double input[N_FFT],
                         vDSP_DFT_Interleaved_SetupD setup,
                         const DSPDoubleComplex twiddle[5][N_FREQ],
                         double power[N_FREQ]) {
    _Alignas(16) DSPDoubleComplex fft_input[80];
    _Alignas(16) DSPDoubleComplex fft_output[5][80];
    for (int r = 0; r < 5; r++) {
        for (int m = 0; m < 80; m++) {
            fft_input[m].real = input[5*m+r];
            fft_input[m].imag = 0.0;
        }
        vDSP_DFT_Interleaved_ExecuteD(setup, fft_input, fft_output[r]);
    }
    for (int k = 0; k < N_FREQ; k++) {
        double re = 0.0, im = 0.0;
        for (int r = 0; r < 5; r++) {
            DSPDoubleComplex a = fft_output[r][k%80];
            DSPDoubleComplex w = twiddle[r][k];
            re += a.real*w.real - a.imag*w.imag;
            im += a.real*w.imag + a.imag*w.real;
        }
        power[k] = re*re + im*im;
    }
}

float *qwen_mel_spectrogram(const float *samples, int n_samples, int *out_frames) {
    if(out_frames)*out_frames=0;
    if(!samples||!out_frames||n_samples<160||n_samples>19200000)return NULL;
    for(int i=0;i<n_samples;i++)if(!isfinite(samples[i]))return NULL;
    int n_fft = N_FFT;
    int n_freqs = N_FREQ;
    int pad_len = n_fft / 2; /* center=True padding (reflect) */

    /* Reflect-pad the signal */
    int padded_len = n_samples + 2 * pad_len;
    float *padded = (float *)malloc(padded_len * sizeof(float));
    if (!padded) return NULL;
    for (int i = 0; i < pad_len; i++) {
        int src = pad_len - i; while(src>=n_samples)src=2*n_samples-2-src; padded[i]=samples[src];
    }
    memcpy(padded + pad_len, samples, n_samples * sizeof(float));
    for (int i = 0; i < pad_len; i++) {
        int src = n_samples - 2 - i; while(src<0||src>=n_samples){if(src<0)src=-src;if(src>=n_samples)src=2*n_samples-2-src;}
        padded[pad_len + n_samples + i] = samples[src];
    }

    int n_frames_total = (padded_len - n_fft) / HOP_LENGTH + 1;
    int n_frames = n_frames_total - 1; /* drop last frame */
    if (n_frames <= 0) {
        fprintf(stderr, "qwen_mel_spectrogram: audio too short (%d samples)\n", n_samples);
        free(padded);
        return NULL;
    }

    double *mel_filters = build_mel_filters();
    if (!mel_filters) { free(padded); return NULL; }
    /* Triangular Slaney filters have compact support. Skip only exactly-zero
     * edges; retain ascending summation order and double precision. */
    int filter_first[N_MEL], filter_end[N_MEL];
    for (int m = 0; m < N_MEL; m++) {
        const double *row = mel_filters + (size_t)m*N_FREQ;
        int first = 0, end = N_FREQ;
        while (first < end && row[first] == 0.0) first++;
        while (end > first && row[end-1] == 0.0) end--;
        filter_first[m] = first;
        filter_end[m] = end;
    }

    /* Periodic Hann window */
    double window[N_FFT];
    for (int i = 0; i < N_FFT; i++)
        window[i] = 0.5f * (1.0f - cos(2.0f * M_PI * (float)i / (float)N_FFT));

    vDSP_DFT_Interleaved_SetupD dft_setup =
        vDSP_DFT_Interleaved_CreateSetupD(NULL, 80, vDSP_DFT_FORWARD,
                                        vDSP_DFT_Interleaved_ComplextoComplex);
    if (!dft_setup) { free(padded); free(mel_filters); return NULL; }
    DSPDoubleComplex twiddle[5][N_FREQ];
    for (int r = 0; r < 5; r++) {
        for (int k = 0; k < N_FREQ; k++) {
            double angle = -2.0*M_PI*(double)r*(double)k/(double)N_FFT;
            twiddle[r][k].real = cos(angle);
            twiddle[r][k].imag = sin(angle);
        }
    }

    /* First pass: compute mel values and find global max.
     * Store as [n_frames, N_MEL] temporarily for convenient max search. */
    float *mel_tmp = (float *)calloc(n_frames * N_MEL, sizeof(float));
    if(!mel_tmp){vDSP_DFT_Interleaved_DestroySetupD(dft_setup);free(padded);free(mel_filters);return NULL;}
    double windowed[N_FFT];
    double power[N_FREQ];
    float global_max = -1e30f;

    for (int t = 0; t < n_frames; t++) {
        int start = t * HOP_LENGTH;
        for (int i = 0; i < N_FFT; i++)
            windowed[i] = padded[start + i] * window[i];

        dft400_power(windowed, dft_setup, twiddle, power);

        for (int m = 0; m < N_MEL; m++) {
            double sum = 0.0f;
            const double *filt = mel_filters + (size_t)m * n_freqs;
            for (int k = filter_first[m]; k < filter_end[m]; k++)
                sum += filt[k] * power[k];
            if (sum < 1e-10f) sum = 1e-10f;
            float val = log10f(sum);
            mel_tmp[t * N_MEL + m] = val;
            if (val > global_max) global_max = val;
        }
    }

    /* Second pass: clamp with dynamic max and normalize.
     * Output layout: [N_MEL, n_frames] for Conv2D compatibility. */
    float *mel = (float *)malloc((size_t)N_MEL * n_frames * sizeof(float));
    if(!mel){free(mel_tmp);vDSP_DFT_Interleaved_DestroySetupD(dft_setup);free(padded);free(mel_filters);return NULL;}
    float min_val = global_max - 8.0f;

    for (int t = 0; t < n_frames; t++) {
        for (int m = 0; m < N_MEL; m++) {
            float val = mel_tmp[t * N_MEL + m];
            if (val < min_val) val = min_val;
            /* Store as [mel_bin, frame] for Conv2D input */
            mel[m * n_frames + t] = (val + 4.0f) / 4.0f;
        }
    }

    free(mel_tmp);
    vDSP_DFT_Interleaved_DestroySetupD(dft_setup);
    free(padded);
    free(mel_filters);

    *out_frames = n_frames;
    return mel;
}
