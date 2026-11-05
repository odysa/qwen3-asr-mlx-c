# Benchmark artifacts

`competitors/`, `q4/` and `head8/` are three separate sessions. Each contains a `TIMING_SUMMARY.json` with per-clip measured times and a manifest and first-measured prediction JSON for each evaluated configuration. `quality/COHORT.json` and the frozen CPU scorer contain the same 20 FLEURS labels used for the published scores. No audio or model weights are redistributed.

Recompute every aggregate error count with Python's standard library:

```sh
python3 results/verify_scores.py
```

The scorer normalizes Unicode with NFKC/casefold, converts punctuation and symbols to spaces, then counts word edits for English and non-space codepoint edits for Chinese. English denominator 212 words; Mandarin 367 characters. Parakeet is predeclared English-only for this experiment; its Chinese rows remain unevaluated. Scores on the same selected cases after tuning are exploratory and do not estimate population accuracy. FLEURS attribution and dataset revision are in `quality/ATTRIBUTION.md`.

All three timing sessions used the same M1 Max, 16 kHz mono PCM16 inputs, one excluded warmup plus three measured repeats per supported model/case, serial model processes, and first measured transcript for scoring. The common warm boundary begins after preparing audio and ends when text is materialized and GPU work completes. Whisper used Metal, four CPU threads, greedy best-of 1 and no temperature fallback. Qwen used greedy decoding and a 256-token cap; Parakeet used BF16 and greedy decoding. Whisper/Parakeet success means the API returned; their adapters did not expose Qwen-style EOS.

The original machine-local seals and full stdout/stderr include local absolute paths and are not included here. The timing summaries are copied from them and references to original plan hashes identify historical receipts, not included public files. `PROVENANCE.json` retains the original plan and pre-run pin hashes with paths removed. This public copy has its own manifest, `MANIFEST.sha256`. Model files, Metal libraries, system state and model auxiliary files must be obtained separately; therefore running the code here is not a byte-for-byte reproduction of those original timing environments. Cold-start-to-first-result was not isolated. Chinese order in the q4/head8 paired sessions was not fully balanced.

`../benchmarks/whisper_resident.cpp` and `../benchmarks/parakeet_resident.py` are the measurement adapters used in the initial competitor session. Whisper.cpp was at commit `4da27c41cebadb3cbbcd2563e9c7cbc8d84bdcad`; Parakeet model revision was `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15` and `parakeet-mlx==0.6.0`. The adapters are provided for inspection; the local orchestration script and binary/model files have machine-specific paths and are not included. Rebuilding them and acquiring FLEURS are required for fresh timing; rerunning the score verifier needs no network or model inference.
