# Performance: complete published measurement inventory

There is no demonstrated state-of-the-art ranking. HTTP serving, developer inference benchmarks and fresh-process measurements have different timing boundaries and warmup schedules. Do not combine their best cells into one framework speedup.

Hardware: Apple M1 Max, 64 GiB unified memory, one shared Mac. FLEURS: first ten English and ten Mandarin test clips, 4.32-15.46 seconds each, 210.66 seconds total. Quality uses 212 English words and 367 Mandarin nonspace codepoints. These previously inspected samples support descriptive/exploratory results, not population equivalence.

## Resident HTTP service: release candidate

Measured source 3826de0; subsequent health and complete-response deadline fixes at a1689ca/6466ef4 have separate CPU and usability validation, not retroactively relabeled timing results. Python 3.14.3, MLX 0.32.4, MLX Audio 0.6.3. Each model starts once; concurrency 1 precedes concurrency 4, with three repeats per clip after one excluded warmup per language per phase. Not every input shape is independently warmed. Native workers use newly compiled graph/frontend/tokenizer objects and existing pinned MLX/weights archives, not a fresh dependency rebuild. No retries.

The p50/p95 values below pool successful client requests, including HTTP upload and response. At concurrency 4 they include queue waiting. Throughput uses successes over the entire measured phase wall time, including failures. The model worker is serialized; concurrent clients are not GPU batching.

| Profile | Clients | EN p50 / p95 ms | ZH p50 / p95 ms | Requests/s | Audio s/s | Failed/total |
|---|---:|---:|---:|---:|---:|---:|
| Native Qwen 0.6B | 1 | 768.8 / 1825.8 | 1038.6 / 2043.4 | 1.02 | 10.69 | 0/60 |
| Native Qwen 0.6B | 4 | 2022.5 / 4234.9 | 4450.2 / 6943.8 | 1.08 | 11.39 | 0/60 |
| Native Qwen 0.6B decoder q8 | 1 | 1097.7 / 1942.5 | 930.3 / 2032.7 | 0.96 | 10.13 | 0/60 |
| Native Qwen 0.6B decoder q8 | 4 | 3184.4 / 5210.0 | 3335.8 / 6769.6 | 1.17 | 12.36 | 0/60 |
| Native Qwen 1.7B | 1 | 1208.8 / 3722.8 | 1587.4 / 3520.7 | 0.56 | 5.85 | 0/60 |
| Native Qwen 1.7B | 4 | 5369.9 / 8629.5 | 8602.4 / 12261.4 | 0.55 | 5.75 | 0/60 |
| MLX Audio Qwen 0.6B | 1 | 755.6 / 1233.6 | 979.7 / 1724.4 | 1.15 | 12.09 | 0/60 |
| MLX Audio Qwen 0.6B | 4 | 2151.8 / 4301.6 | 4354.7 / 5867.8 | 1.16 | 12.17 | 0/60 |
| MLX Audio Whisper tiny | 1 | 268.3 / 642.7 | 448.5 / 882.4 | 2.51 | 26.40 | 0/60 |
| MLX Audio Whisper tiny | 4 | 1103.4 / 1793.2 | 2094.5 / 2612.9 | 2.46 | 25.89 | 0/60 |
| MLX Audio Parakeet v3 | 1 | 469.7 / 742.2 | N/A | 2.18 | 19.87 | 0/30 |
| MLX Audio Parakeet v3 | 4 | 728.9 / 2016.8 | N/A | 4.10 | 37.30 | 0/30 |

### Quality, startup and memory

Readiness is one process-launch-to-HTTP-ready observation, not a distribution. Native weights may remain lazy until the first request; MLX Audio explicitly evaluates weights before readiness. The first excluded request is retained in the warmup JSON. Neither readiness value is cold-to-first-transcript latency.

RSS is the process-lifetime peak after concurrency 4, including initialization, warmups and the preceding concurrency-1 phase. Parent and native-child peaks are separate; their sum is not measured simultaneous memory.

| Profile / stored weight format | Ready s (one sample) | EN WER | ZH CER | Parent RSS MiB | Native worker RSS MiB |
|---|---:|---:|---:|---:|---:|
| Native Qwen 0.6B / BF16 checkpoint; native graph | 0.843 | 10/212 (4.72%) | 38/367 (10.35%) | 47.9 | 1560.1 |
| Native Qwen 0.6B decoder q8 / Affine decoder q8/group64; remaining BF16 | 0.828 | 10/212 (4.72%) | 38/367 (10.35%) | 45.6 | 1181.6 |
| Native Qwen 1.7B / BF16 checkpoint; native graph | 0.342 | 6/212 (2.83%) | 35/367 (9.54%) | 43.2 | 3959.4 |
| MLX Audio Qwen 0.6B / BF16 checkpoint; default MLX Audio execution | 5.017 | 10/212 (4.72%) | 38/367 (10.35%) | 1881.1 | N/A |
| MLX Audio Whisper tiny / F16 checkpoint; default MLX Audio execution | 5.290 | 32/212 (15.09%) | 135/367 (36.78%) | 542.7 | N/A |
| MLX Audio Parakeet v3 / F32 checkpoint tensors; default MLX Audio execution (no BF16 cast) | 4.617 | 10/212 (4.72%) | N/A | 2274.7 | N/A |

The frozen Parakeet run ID incorrectly contains "bf16". Its checkpoint has 697 F32 tensors; the generic MLX Audio loader does not apply the BF16 cast used by the older parakeet-mlx benchmark. Here it is reported as F32/default MLX Audio execution. Stored precision does not imply identical activation precision.

Complete responses retain inference, service and queue timing; SCORES_AND_METRICS.json includes inference p50 and queue p95. Native EOS/token IDs are retained; MLX Audio completion means its generator returned. Failed/capped first outputs remain failures under the frozen scorer. The tables use the first measured transcript for quality, with any repeat changes listed in JSON. See [serving timing semantics](../docs/SERVING.md).

### Earlier API development run, retained separately

The e5f3790 service lacked pre-upload connection admission, fixed before release. The three-model run used the same language warmup and repeat schedule. It remains in serving-v0. The substantial session-to-session variation is unresolved; no code-induced speedup or regression is inferred from comparing these sessions. After the later run, a host snapshot showed load average around 91 and OrbStack around 725% CPU; a further timestamped post-run snapshot is retained in serving-v1/POST_RUN_LOAD.json. These snapshots do not establish continuous load during every request or prove the cause of all differences. Treat serving-v1 as shared-host stress observations, not isolated hardware capacity. A quiet-window comparison remains outstanding.

| Earlier profile | Clients | Requests/s | EN client p50 ms | ZH client p50 ms | Failed/total |
|---|---:|---:|---:|---:|---:|
| MLX Audio Parakeet v3 | 1 | 11.84 | 89.6 | N/A | 0/30 |
| MLX Audio Parakeet v3 | 4 | 12.70 | 315.0 | N/A | 0/30 |
| MLX Audio Qwen 0.6B | 1 | 4.01 | 233.6 | 229.7 | 0/60 |
| MLX Audio Qwen 0.6B | 4 | 4.00 | 945.0 | 1066.3 | 0/60 |
| MLX Audio Whisper tiny | 1 | 8.23 | 108.3 | 127.6 | 0/60 |
| MLX Audio Whisper tiny | 4 | 8.45 | 424.4 | 521.9 | 0/60 |

## Historical inference comparison: a different boundary

These November 4-5 sessions exclude model initialization and WAV reading, and end with synchronized GPU work and materialized text. Each clip has its own excluded warmup and three measured runs in a separate process. Central latency is the median of per-clip medians, not the HTTP table's pooled client p50. P95 below is recomputed over each language's 30 saved inference times. RSS is the maximum per-process peak across clips, including setup and warmup. This is not service throughput or streaming latency.

| Session / implementation | EN median / pooled p95 ms | ZH median / pooled p95 ms | EN WER | ZH CER | Max RSS GiB |
|---|---:|---:|---:|---:|---:|
| competitors / qwen-bf16 | 198.6 / 294.3 | 192.7 / 364.2 | 10/212 (4.72%) | 38/367 (10.35%) | 1.53 |
| competitors / qwen-q8 | 172.5 / 249.6 | 168.7 / 311.2 | 10/212 (4.72%) | 38/367 (10.35%) | 1.15 |
| competitors / whisper-small | 176.4 / 231.2 | 230.1 / 463.8 | 15/212 (7.08%) | 55/367 (14.99%) | 0.73 |
| competitors / whisper-turbo | 429.0 / 466.0 | 489.3 / 774.8 | 10/212 (4.72%) | 30/367 (8.17%) | 1.81 |
| competitors / parakeet-v3 | 124.2 / 161.2 | N/A | 10/212 (4.72%) | N/A | 1.90 |
| q4 / qwen-q8 | 173.6 / 249.4 | 171.6 / 312.2 | 10/212 (4.72%) | 38/367 (10.35%) | 1.15 |
| q4 / qwen-q4 | 160.8 / 226.1 | 163.8 / 287.2 | 11/212 (5.19%) | 40/367 (10.90%) | 0.94 |
| q4 / parakeet-v3 | 124.4 / 159.5 | N/A | 10/212 (4.72%) | N/A | 1.97 |
| head8 / qwen-q8 | 171.6 / 250.2 | 170.3 / 312.5 | 10/212 (4.72%) | 38/367 (10.35%) | 1.15 |
| head8 / qwen-head8 | 161.6 / 232.8 | 162.1 / 289.9 | 10/212 (4.72%) | 38/367 (10.35%) | 1.30 |
| head8 / parakeet-v3 | 128.2 / 168.0 | N/A | 10/212 (4.72%) | N/A | 2.10 |

All 90 competitor jobs, 50 q4 jobs and 50 head8 jobs completed, without retries. Q4 added one English word error and two Mandarin character errors; head8 preserved the 20 observed texts but added about 165 MB of process peak RSS. Neither caught same-session Parakeet English latency. The separate q2 smoke reached its token cap without EOS and was rejected; it has no completed quality or speed score. Original provenance and limitations: [artifact guide](README.md).

## Fresh-process and repeated-file measurements

The [BF16 0.6B/1.7B process smoke](multimodel-bf16.md) used two official examples without pinned reference transcripts, three fresh processes per cell, retained filesystem caches, and process launch through exit:

| Native model | EN process median s | ZH process median s | Max process RSS GiB |
|---|---:|---:|---:|
| Qwen 0.6B BF16 | 0.585 | 0.336 | 1.53 |
| Qwen 1.7B BF16 | 1.099 | 0.561 | 3.87 |

These are not resident warm latency or accuracy measurements. In a separate two-trial Whisper tiny repeated-file test, three requests took 3.27/3.19 s in one model process versus 10.58/8.99 s in three processes. Order was fixed, separate processes first, and caches/model state persisted. That measures total process time, not per-request model latency; memory and accuracy were not evaluated.

## Coverage and limits

[Model compatibility](asr-compatibility.md) records each checkpoint, language scope and license. Smoke-only families have no comparative speed, throughput, memory or quality score here; missing is not zero. These service tests cover six profiles only, not every compatible checkpoint.

Streaming partial/final latency, first token/audio, endpointing, long recordings, noisy/far-field speech, diarization correctness, simultaneous multi-model residency, second-machine scaling and controlled comparisons with complete serving frameworks are not measured. CPU lifecycle tests cover upload admission, queue overload/recovery, HTTP errors and native pipe reuse; those are not inference-quality evidence.

The data support a working resident local framework and explicit model/precision tradeoffs, not a universal speed or quality lead.

## Verify without downloading models

    python3.14 results/verify_scores.py
    python3.14 tools/summarize_serving.py --results results/serving-v0 --check
    python3.14 tools/summarize_serving.py --results results/serving-v1 --check
    python3.14 -m unittest discover -s tests -v

Every serving phase retains hashes, request responses, warmups, first-repeat predictions, metrics and summaries. Public data are derived exports with private paths removed and new manifests; original private plans and audio/weights are not bundled. Artifact hashes identify measurements, not a guarantee of bitwise-identical rebuilds. Instructions for fresh measurements are in the serving guide.
