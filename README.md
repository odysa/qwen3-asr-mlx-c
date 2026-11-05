# Qwen3-ASR on Apple Silicon: native MLX C inference

A local C11 application for Qwen3-ASR 0.6B and 1.7B on Apple Silicon, built on the official MLX C API and Metal. The default model path uses BF16. For 0.6B only, separate experimental builds quantize decoder linear weights to q8 or q4, or additionally use a q8 output head while retaining BF16 input embeddings. Model weights and audio are **not** included.

## Local results

On an Apple M1 Max, we compared completed warm requests on the first 10 English and first 10 Mandarin test rows from FLEURS. One warmup and three timed requests per model/clip; numbers below are medians across the per-clip medians. Error rates are corpus WER/CER on the first timed transcript. Models, architectures and numerical precision differ.

| Configuration | English latency / WER | Mandarin latency / CER |
|---|---:|---:|
| Qwen decoder q8 | 172 ms / 4.72% | 169 ms / 10.35% |
| Whisper small FP16 (whisper.cpp/Metal) | 176 ms / 7.08% | 230 ms / 14.99% |
| Whisper large-v3-turbo FP16 (whisper.cpp/Metal) | 429 ms / 4.72% | 489 ms / 8.17% |
| Parakeet TDT v3 BF16 (parakeet-mlx) | 124 ms / 4.72% | Not evaluated |

Qwen q8 was faster on all Chinese clips than Whisper small and made fewer errors; English speed versus small was effectively tied (five clips faster each). Qwen was faster than turbo on all 20 clips, but turbo made fewer Chinese errors. Parakeet was faster on all English clips with the same aggregate English error count. **This is a 20-clip, single-machine diagnostic, not a general model ranking.**

Two later exploratory sessions reused the same clips:

- Decoder q4 versus decoder q8: English 161 vs 174 ms; Mandarin 164 vs 172 ms. Q4 added one English word error (11/212 vs 10/212) and two Chinese character errors (40/367 vs 38/367).
- Decoder q8 plus q8 output head versus decoder q8: English 162 vs 172 ms; Mandarin 162 vs 170 ms. All 20 first-measured texts matched q8, but broader quality equivalence is untested. The extra head adds approximately 165 MB of weight data and the English median still trails the same-session Parakeet result (128 ms).

The 20 clips had already been inspected across runs. Reported inference time excludes model initialization and WAV reading; full cold-process startup was not isolated. Process-lifetime RSS includes setup and warmup. The published summaries are sanitized exports, not byte-identical copies of the original machine-local sealed bundles. See [results/README.md](results/README.md) for methodology, per-clip measurements, scorer and limitations.

## Build on macOS

Requires Apple Silicon, Xcode with Metal toolchain, CMake, and Git. The dependency script fetches the pinned `ml-explore/mlx-c` commit; its CMake build fetches its pinned MLX dependency.

```sh
tools/fetch_deps.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQWEN_EXPERIMENTS=ON
cmake --build build --target qwen3-asr qwen3-asr-q8 qwen3-asr-q4 qwen3-asr-head8 -j4
python3 tools/fetch_model.py
```

`fetch_model.py` downloads the original public checkpoint and example WAVs. This is a large download; inspect and retain its generated manifest. The Chinese example is 16 kHz mono PCM16. For another file, prepare a 16 kHz mono PCM16 WAV; the current implementation accepts approximately 30 seconds per segment and uses greedy decoding with a 256-token cap.

```sh
./build/qwen3-asr transcribe --model models/Qwen3-ASR-0.6B \
  --audio fixtures/asr_zh.wav --language Chinese --json
```

### Larger BF16 model

The same native executable also runs the official 1.7B checkpoint, automatically loading its two safetensors shards. This is speech recognition; speech generation and conversational audio are not implemented. Downloading model weights requires network access, but inference is local. The 1.7B variant has only been smoke-tested; the latency and quality table above covers 0.6B only.

```sh
python3 tools/fetch_model.py --size 1.7B
./build/qwen3-asr transcribe --model models/Qwen3-ASR-1.7B \
  --audio fixtures/asr_zh.wav --language Chinese --json
```

Experimental q8, q4 and head8 binaries and the conversion tool support 0.6B only; use the BF16 executable for 1.7B.

## Experimental quantized paths

Conversion is local and requires Python with MLX (`mlx==0.32.4` used for the reported experiments). It writes a new model directory, refuses to overwrite an existing one, and records source/output hashes. Q8 and q4 each quantize the same 196 decoder linear matrices; the head8 conversion adds a quantized output-head copy to the q8 checkpoint. It does not train or improve the model, and precision changes can alter transcripts.

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install 'mlx==0.32.4'
python tools/convert_model.py --kind q8 --source models/Qwen3-ASR-0.6B --output experimental/q8-model
python tools/convert_model.py --kind q4 --source models/Qwen3-ASR-0.6B --output experimental/q4-model
python tools/convert_model.py --kind head8 --source experimental/q8-model --output experimental/head8-model
./build/qwen3-asr-q8 transcribe --model experimental/q8-model --audio fixtures/asr_zh.wav --language Chinese --json
./build/qwen3-asr-head8 transcribe --model experimental/head8-model --audio fixtures/asr_zh.wav --language Chinese --json
```

If MLX cannot locate its Metal library, set `MLX_METAL_PATH` to the generated `mlx.metallib` in the CMake build tree. Q8/q4/head8 are experimental and do not change the BF16 default. The resident timing binaries are separate CMake targets: `qwen3-asr-q8-resident`, `qwen3-asr-q4-resident`, and `qwen3-asr-head8-resident`.

## Attribution and scope

FLEURS labels are redistributed under CC BY 4.0; see [attribution](results/quality/ATTRIBUTION.md). The adapted tokenizer and audio front end retain the upstream MIT notices; MLX Audio and Qwen reference licenses are in `licenses/`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for project dependencies and provenance. The Qwen checkpoint is obtained from its own upstream repository under its terms. No blanket license for other original files is specified here.
