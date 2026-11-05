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

The same native executable also runs the official 1.7B checkpoint, automatically loading its two safetensors shards. This is speech recognition; speech generation and conversational audio are not implemented. Downloading model weights requires network access, but inference is local. The 1.7B variant has only been smoke-tested; the latency and quality table above covers 0.6B only. See [the separate fresh-process smoke comparison](results/multimodel-bf16.md), which is not an accuracy or warm-latency test.

```sh
python3 tools/fetch_model.py --size 1.7B
./build/qwen3-asr transcribe --model models/Qwen3-ASR-1.7B \
  --audio fixtures/asr_zh.wav --language Chinese --json
```

Experimental q8, q4 and head8 binaries and the conversion tool support 0.6B only; use the BF16 executable for 1.7B.

## Local voice interface (experimental)

One entry point supports native Qwen3-ASR transcription plus optional Parakeet transcription and Qwen3-TTS / Kokoro speech generation. Parakeet uses [parakeet-mlx](https://github.com/senstella/parakeet-mlx); the TTS models use [MLX Audio](https://github.com/Blaizzy/mlx-audio). The TTS backend is MLX Audio, not this repository's native C graph. These are separate speech-to-text and text-to-speech tasks; conversational speech-to-speech is not yet implemented.

After building the C executable and downloading an ASR checkpoint as above:

```sh
python3 voice/voice.py transcribe --model models/Qwen3-ASR-0.6B \
  --audio fixtures/asr_zh.wav --language Chinese
```

For the optional Qwen3-TTS 0.6B CustomVoice 4-bit model, create a separate Python environment and download the pinned community conversion once (check that model's terms):

```sh
python3 -m venv .venv-voice && . .venv-voice/bin/activate
python -m pip install -r voice/requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit', revision='08c72cad5e2fd0f41730c8bd1f28149585e46361', local_dir='models/Qwen3-TTS-0.6B-CustomVoice-4bit')"
python voice/voice.py speak --model models/Qwen3-TTS-0.6B-CustomVoice-4bit \
  --text 'Hello from my Mac.' --speaker Vivian --language English --output hello.wav
```

The `speak` inference command requires a local model directory and sets Hugging Face offline mode. It writes a new WAV, refuses to overwrite an existing file, and returns JSON with the sample rate and duration. If writing fails, a partial output may remain; inspect or remove it and choose a new output path for a retry. The optional dependency versions shown are those used for the initial smoke test; transitive dependencies are not locked. One English example produced a nonempty 24 kHz WAV that the ASR path transcribed correctly except punctuation. This is a functional smoke test, not a speech-quality or latency benchmark. The TTS model and its dependencies have their own licenses and are downloaded separately.

### Non-Qwen models (experimental)

The Parakeet TDT v3 adapter and Kokoro 82M adapter have each run an English example locally. Parakeet returns when its generator finishes; this is not an exposed EOS check. The adapter accepts `auto` or `English` for Parakeet; other languages are outside this integration's tested scope. Kokoro is restricted to American or British English (`a` or `b`) in this adapter. These examples do not establish speech quality or comparative speed.

The following setup was tested in a clean Python 3.14 environment and with Apple Silicon Homebrew at `/opt/homebrew`. It installs the English dependencies directly because the `misaki[en]` extra does not resolve reliably on Python 3.14. It downloads model files and the spaCy language package during setup. The inference commands use local paths with Hugging Face offline mode. Kokoro requires the Homebrew espeak-ng library/data: the tested `espeakng-loader` wheel refers to a build-machine path, so the adapter explicitly selects Homebrew's installation. The Python package versions listed are direct dependencies; their transitive dependencies are not locked.

```sh
python3.14 -m venv .venv-voice314 && . .venv-voice314/bin/activate
python -m pip install -r voice/requirements-non-qwen.txt
python -m spacy download en_core_web_sm
brew install espeak-ng
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/parakeet-tdt-0.6b-v3', revision='ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15', local_dir='models/parakeet-tdt-0.6b-v3')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Kokoro-82M-bf16', revision='a71e4d38b236d968966a2002c4c895dbd12b1c3c', local_dir='models/Kokoro-82M')"
python voice/voice.py transcribe --backend parakeet --model models/parakeet-tdt-0.6b-v3 \
  --audio fixtures/asr_en.wav --language English
python voice/voice.py speak --backend kokoro --model models/Kokoro-82M \
  --text 'Hello from my Mac.' --speaker af_heart --language a --output kokoro.wav
```

Parakeet requires an audio decoder supported by its backend; the tested example was 16 kHz mono PCM16. The Kokoro model repository's `bf16` name is not a guarantee about the dtype of every stored tensor. Both models and dependencies are distributed separately under their own terms. If Kokoro's spaCy package or Homebrew data is absent, the command fails rather than attempting a download during inference. Missing Kokoro voice assets may resolve from an existing local Hugging Face cache; this setup is not a self-contained offline installer.

### Generic local MLX Audio ASR

The `mlx-audio` ASR backend attempts any local checkpoint that the installed MLX Audio version can load and whose `generate` method returns a single result with a text field. The model-specific API, dependencies, options and licensing still apply. This interface does not establish that every installed model family or checkpoint works, and it does not support streaming or forced alignment. We have smoke-tested local Whisper tiny, Parakeet TDT v3, Qwen3-ASR 0.6B, Moonshine tiny, SenseVoiceSmall and Fun-ASR Nano; the dedicated Qwen C and Parakeet paths above remain available.

```sh
# Python 3.14 ASR environment (separate from the older Kokoro setup).
python3.14 -m venv .venv-asr314 && . .venv-asr314/bin/activate
python -m pip install -r voice/requirements.txt
python voice/voice.py list-asr
# One explicit example, downloaded once at a pinned revision.
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/whisper-tiny-asr-fp16', revision='77fa3f52b482ec80d086df55f893065a9baab172', local_dir='models/whisper-tiny-asr-fp16')"
python voice/voice.py transcribe --backend mlx-audio --model models/whisper-tiny-asr-fp16 \
  --audio fixtures/asr_en.wav --language en
# Reuse a loaded model across multiple sequential requests (not parallel batching).
python voice/voice.py transcribe-many --model models/whisper-tiny-asr-fp16 \
  --audio fixtures/asr_en.wav fixtures/asr_zh.wav --options-json '{"temperature":0}'
```

`list-asr` lists modules installed in MLX Audio, not verified model support. `--options-json` passes model-specific generation arguments; `audio`, `language`, and `stream` cannot be overridden there. With `--language auto`, the backend's default is used; if a language is specified the model must expose that parameter. A returned `status: ok` means the generator returned a text result, not that an EOS token was checked or the transcription is correct. Inference uses a local directory and requests Hugging Face offline mode; third-party dependencies may require their own offline setup. `transcribe-many` retains model state across files and aborts without partial JSON if any file fails; equivalence to separate model loads has not been established for arbitrary backends.

The multi-file mode avoids repeated process startup and model loading. In an exploratory M1 Max test of Whisper tiny on the same short English clip three times, two end-to-end trials took 3.27/3.19 seconds in one process versus 10.58/8.99 seconds in three separate processes. All returned texts matched. Each trial used the fixed order of separate processes first and the reused-model process second. This compares total process wall time, with filesystem caches retained, not model-only latency; no memory or transcription accuracy was measured. Full local runner and raw-output receipt seal SHA-256: `ff4085a2aeb50ee037af9c6a689d3f6f0ab70d8aa197635d78c452d15ee615b2` (machine-local receipt not published). Code used for the timing was originally recorded as `849fe68`; after the author-attribution rewrite its equivalent is `1c1b5da` (identical source tree). this small test is exploratory; the raw receipts and timing arithmetic were independently reviewed without rerunning inference.

### Additional locally tested ASR families

Python 3.14 with `mlx-audio==0.6.3` ran a short example on the following pinned checkpoints. These are smoke tests on this Mac, not quality, speed or coverage guarantees. See [ASR compatibility](results/asr-compatibility.md) for recorded download revisions, scope and model licenses. The first seven model files were not hashed at measurement time, so their receipts do not bind their bytes to those revisions; the eight later models were hashed before and after their runs.

| Family / checkpoint | CLI backend | Example language |
|---|---|---|
| Moonshine tiny | `mlx-audio` | English |
| SenseVoiceSmall | `mlx-audio` | Mandarin |
| Fun-ASR Nano | `mlx-audio` | Mandarin |
| wav2vec2 base-960h CTC | `wav2vec2-ctc` | English |
| Nemotron 3.5 streaming 0.6B 8-bit | `mlx-audio` | English, whole file |
| Canary 1B v2 q8 | `mlx-audio` | English, whole file |
| Granite 4.0 1B Speech | `mlx-audio` | English, whole file |
| VibeVoice-ASR 4-bit | `mlx-audio` | English; structured JSON returned as text |
| Voxtral Mini Realtime 4-bit | `mlx-audio` | English, whole file |
| FireRedASR2-AED | `mlx-audio` | Mandarin, whole file |
| MMS 1B FL102, English adapter | `mlx-audio` | English only; non-commercial |
| Distil-Whisper small.en | `mlx-audio` | English; Whisper derivative |

For example, the two additional checkpoints can be downloaded and run as follows (the Nemotron example is a whole-file call, not a streaming API):

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit', revision='7279359e4481b5e9e185a318bd618e429c6d86cd', local_dir='models/nemotron-streaming-8bit')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Mediform/canary-1b-v2-mlx-q8', revision='0b6b32ee10f30c89e3ead7249bb636445e3019ee', local_dir='models/canary-1b-v2-q8')"
python voice/voice.py transcribe --backend mlx-audio --model models/nemotron-streaming-8bit --audio fixtures/asr_en.wav --language en-US
python voice/voice.py transcribe --backend mlx-audio --model models/canary-1b-v2-q8 --audio fixtures/asr_en.wav --options-json '{"source_lang":"en","target_lang":"en"}'
```

For IBM Granite Speech (a larger download), the tested whole-file example is:

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-4.0-1b-speech', revision='bd87ab862416353633ea431fe49b1614003623c5', local_dir='models/granite-4.0-1b-speech')"
python voice/voice.py transcribe --backend mlx-audio --model models/granite-4.0-1b-speech --audio fixtures/asr_en.wav
```

The VibeVoice conversion has no tokenizer files. Download its model and the separate pinned Qwen2.5 tokenizer, then copy only the four tokenizer files into the model directory. Without those files the offline run fails on a machine with no matching local cache. The `text` field is a JSON string; the wrapper does not parse or validate the returned speaker or timestamp fields.

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/VibeVoice-ASR-4bit', revision='a1a15cb6c7b70f76b588af7e12f6fab34d5ab654', local_dir='models/VibeVoice-ASR-4bit')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B', revision='d149729398750b98c0af14eb82c78cfe92750796', local_dir='models/VibeVoice-tokenizer', allow_patterns=['tokenizer.json','tokenizer_config.json','merges.txt','vocab.json'])"
cp models/VibeVoice-tokenizer/{tokenizer.json,tokenizer_config.json,merges.txt,vocab.json} models/VibeVoice-ASR-4bit/
python voice/voice.py transcribe --backend mlx-audio --model models/VibeVoice-ASR-4bit --audio fixtures/asr_en.wav --options-json '{"max_tokens":1024,"temperature":0}'
```

For the tested Voxtral and FireRed checkpoints, use the generic adapter (the Voxtral command is whole-file, not a streaming test):

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit', revision='fdebf7b2af834a1db4b8a3c99ab7480b333adf9e', local_dir='models/Voxtral-Mini-Realtime-4bit')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/FireRedASR2-AED-mlx', revision='f3212eacfa49b851130b97c63653c8e06ee09bdb', local_dir='models/FireRedASR2-AED-mlx')"
python voice/voice.py transcribe --backend mlx-audio --model models/Voxtral-Mini-Realtime-4bit --audio fixtures/asr_en.wav
python voice/voice.py transcribe --backend mlx-audio --model models/FireRedASR2-AED-mlx --audio fixtures/asr_zh.wav
```

MMS is a non-commercial checkpoint; this example downloads only its English adapter. The wrapper does not expose tested selection of other MMS language adapters. Distil-Whisper is a Whisper derivative:

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/mms-1b-fl102', revision='d483345545bea550895b1aa0c6ba40236b9f1e22', local_dir='models/mms-1b-fl102-eng', allow_patterns=['*.json','*.txt','*.model','model.safetensors','adapter.eng.safetensors'])"
python -c "from huggingface_hub import snapshot_download; snapshot_download('distil-whisper/distil-small.en', revision='9e4a67ca4569c30be43a3fe7fba1621e504f0093', local_dir='models/distil-small.en', allow_patterns=['*.json','*.txt','*.model','*.tiktoken','model.safetensors'])"
python voice/voice.py transcribe --backend mlx-audio --model models/mms-1b-fl102-eng --audio fixtures/asr_en.wav
python voice/voice.py transcribe --backend mlx-audio --model models/distil-small.en --audio fixtures/asr_en.wav
```

The dedicated wav2vec2 CTC adapter uses MLX Audio's encoder, normalizes the waveform, runs the checkpoint's CTC projection and greedy decoder, and currently accepts only the checked preprocessing/vocabulary shape. Its outputs matched the Hugging Face PyTorch implementation on three short English clips. This does not establish equivalent accuracy for other checkpoints or audio. The adapter does not provide timestamps, streaming or an exposed EOS. It requires a local `model.safetensors`, vocabulary and config:

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/wav2vec2-base-960h', revision='22aad52d435eb6dbaf354bdad9b0da84ce7d6156', local_dir='models/wav2vec2-base-960h', allow_patterns=['*.json','*.safetensors','*.txt'])"
python voice/voice.py transcribe --backend wav2vec2-ctc --model models/wav2vec2-base-960h \
  --audio fixtures/asr_en.wav --language English
```

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
