# Resident local ASR API

The framework offers two serving engines behind the same HTTP interface:

- **Native:** the repository's C/MLX graph, with a persistent subprocess holding Qwen weights and tokenizer state. BF16 supports Qwen3-ASR 0.6B and 1.7B. Experimental q8/q4/head8 workers support 0.6B only.
- **MLX Audio:** a resident optional Python/MLX backend for local model directories whose generator returns one text result. This retains the existing model adapters' checkpoint-specific limitations. A module being installed is not proof that every checkpoint works.

Both engines run locally on Apple Silicon. Each process serves one loaded model. Request concurrency is handled by a queue; it is not GPU batching, streaming recognition, speech-to-speech, or dynamic multi-model scheduling.

## Start a native worker

Use Python 3.14. The HTTP layer itself uses only the standard library; native inference requires the built MLX C worker and a downloaded checkpoint, not Python MLX.

```sh
tools/fetch_deps.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQWEN_EXPERIMENTS=ON
cmake --build build --target qwen3-asr-worker qwen3-asr-q8-worker -j4
# First download/convert the model using the instructions in README.md.
python3.14 voice/voice.py serve --backend native \
  --executable build/qwen3-asr-worker \
  --model models/Qwen3-ASR-0.6B --model-id qwen-bf16 --port 8080
```

For q8, use `build/qwen3-asr-q8-worker` with the locally converted q8 checkpoint. The q4/head8 worker targets follow the same naming scheme. A worker must match its checkpoint format. For 1.7B, use the BF16 worker. Native decoding is greedy with a 256-token cap; a capped/failed decode is an error rather than a successful transcript. KV state is reset after each completed request and after all outstanding GPU work synchronizes. A worker that fails or exits requires a service restart; /health then returns 503 rather than readiness. Failed jobs are counted, never silently retried.

If the MLX C build cannot locate its Metal library, set `MLX_METAL_PATH` to the generated `mlx.metallib`, as described in README.md.

## Start an MLX Audio model

```sh
python3.14 -m venv .venv-asr
. .venv-asr/bin/activate
python -m pip install -r voice/requirements.txt
python voice/voice.py serve --backend mlx-audio \
  --model models/whisper-tiny-asr-fp16 --model-id whisper-tiny --port 8080
```

The model directory must already exist. The loader requests Hugging Face offline mode; model-specific tokenizer/assets and other third-party dependencies must also be prepared locally. Model loading and inference happen on the same single worker thread. MLX Audio weights are explicitly evaluated before readiness; native C weights may remain lazy until the first request. Readiness and first-request time are therefore different metrics.

`--options-json` fixes model-specific generation settings at startup, for example `'{"max_tokens":256,"temperature":0}'` for Qwen. Options cannot override `audio`, `language` or `stream`. Language values are backend-specific: native accepts `auto`, `English`/`en` and `Chinese`/`zh`; Whisper expects `en`/`zh`; models without a language argument require `auto`. Do not assume the same language API for every architecture.

## Transcribe an uploaded WAV

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@recording-16k-mono.wav -F language=auto
curl http://127.0.0.1:8080/metrics
```

The transcription endpoint follows the common multipart transcription shape, not every feature of an OpenAI API/SDK. Accepted fields are `file`, optional `model` (must equal the loaded ID), `language`, and `response_format=json|text`. It does not accept server filesystem paths or URLs. The service validates uncompressed mono PCM16 WAV at 16 kHz; it does not silently resample audio. Default limits are 30 seconds of audio and 16 MiB per upload. Backend-specific model length limits still apply if you raise the HTTP limit.

A JSON result contains `text`, `model`, completion evidence and:

- `queue_s`: time waiting after upload validation and admission to the GPU queue.
- `inference_s`: engine invocation through materialized text and synchronization. Native mode includes pipe exchange and worker WAV reading; MLX Audio includes the generator's file preprocessing. Neither includes HTTP upload or temporary-file creation.
- `service_s`: upload-validated queue admission through temporary-file cleanup and the engine result; it excludes upload/parsing and response transmission.
- `audio_s` and `real_time_factor`: audio duration and inference seconds divided by audio seconds; lower is faster.

The client-observed time additionally includes upload, multipart parsing, response encoding and transport. Native responses expose EOS and token IDs. MLX Audio's `generate_returned` means its generator returned text, not that this wrapper independently verified EOS. If a configured generation limit is reached according to an exposed token count, `possibly_capped` is returned; the benchmark treats that as a failed quality output. Not all backends expose equivalent token counts or termination evidence.

## Capacity, failures and metrics

The service binds only to `127.0.0.1`, rejects browser Origin requests, and is intended for trusted local clients. It provides no authentication/TLS or remote deployment mode. Do not expose it through a public proxy as-is.

`--max-connections` defaults to 8 and bounds HTTP threads **before** reading upload bodies. Additional connections get HTTP 503. Each body has a byte cap and a 15-second socket inactivity timeout. `--max-pending` defaults to 4 queued jobs in addition to one active inference; excess admitted uploads receive HTTP 429. Limits bound resource use, not a promised latency SLA. Admitted jobs may finish even if their clients disconnect. The MLX Audio path does not forcibly interrupt a running generator; choose model-specific caps and limits accordingly. The native bridge has a 120-second worker-response deadline and terminates an unresponsive worker.

`/metrics` reports accepted model jobs, completed/failed jobs, inference-queue rejections, connection-admission rejections, pending jobs, successful service p50/p95, queue p95, completed audio seconds and process-lifetime peak RSS. Latency history retains the latest 10,000 successful requests; counters are lifetime. Invalid uploads never become model jobs. Parent process peak RSS and native-worker peak RSS are separate: their peaks can occur at different times and must not be presented as a measured simultaneous total. Python/MLX Audio inference runs in the parent process. All peaks include setup and warmup.

## Reproduce an HTTP benchmark

See [the performance report](../results/PERFORMANCE.md) for the retained results and timing boundaries.

```sh
python3.14 tools/benchmark_server.py --url http://127.0.0.1:8080 \
  --cohort results/quality/COHORT.json --audio-root /path/to/pinned-audio-root \
  --configs en_us cmn_hans_cn --languages-json '{"en_us":"en","cmn_hans_cn":"zh"}' \
  --repeat 3 --concurrency 1 --output /tmp/voice-bench-c1
```

Audio is not bundled. The cohort records exact audio paths/hashes; supply matching WAVs or your own manifest with `rows` containing `case_id`, `config`, `audio_path`, `audio_sha256` and `duration_s`. The tool refuses hash mismatches and existing output directories, does no downloads and never retries requests. It retains warmups, individual HTTP responses/errors, first-repeat predictions, metrics, summaries and hashes. It records empty/non-JSON errors, including overload 503, instead of aborting or dropping them. Failed requests remain in request counts; latency percentiles are successful-request distributions. Requests/s uses successes divided by the entire measured phase wall time, including failures. Audio seconds/s uses successfully completed audio over that wall time.

The included FLEURS reference labels and frozen WER/CER normalization apply only to that cohort. Your own audio needs its own reference labels and quality evaluation. The first timed transcript is selected before scoring; later-repeat changes are reported. Running more clients queues requests on the same worker and may increase tail latency without increasing throughput.
