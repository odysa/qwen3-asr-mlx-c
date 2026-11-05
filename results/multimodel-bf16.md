# Qwen3-ASR 0.6B and 1.7B: local process smoke measurements

On an Apple M1 Max, each measurement launches a fresh BF16 C/MLX executable, loads the model, reads one official example WAV, transcribes it, and exits. Values are the median of three sequential process-wall measurements for each model and clip. Filesystem caches were not cleared. RSS is the maximum process peak observed across those runs, including loading. These are **not** resident warm-inference latencies.

| Model | English process time | Chinese process time | Max process RSS |
| --- | ---: | ---: | ---: |
| Qwen3-ASR 0.6B BF16 | 0.585 s | 0.336 s | 1.53 GiB |
| Qwen3-ASR 1.7B BF16 | 1.099 s | 0.561 s | 3.87 GiB |

Both examples returned `status: ok`; the clips have no pinned reference transcripts and these results do not measure accuracy. The 0.6B output matched the previously shipped implementation on both examples. This is one machine, two clips, and three fresh processes per cell, not a general performance ranking.

The code tested was originally recorded as commit `d8f4ea8`; after the author-attribution rewrite its equivalent is `8747b77` (identical source tree). official checkpoint revisions are `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` (0.6B) and `7278e1e70fe206f11671096ffdd38061171dd6e5` (1.7B). Input files and the executable were hashed before and after the run. An independent reviewer checked all 26 receipt files, their 19 input hashes, raw outputs, and the arithmetic. The original local receipt bundle's `SEAL.json` SHA-256 is `e2427f93423e2e408b75f1a56544f2ec9ad5a2f354e134258b7319dba13126f6`; it contains machine-local paths and is not distributed here.

An earlier twelve-row `benchmarks/multimodel-smoke.json` file is retained as **unverified exploration**: its runner, raw stderr and measured binary/input hashes were not retained. It is not the source of the numbers above.

The loader and graph review covered the inspected official BF16 checkpoints. It does not validate arbitrary model configurations or corrupted/cached downloads. Experimental q8/q4/head8 paths remain 0.6B only.
