#!/usr/bin/env python3
"""Local voice CLI: native Qwen ASR and optional MLX Audio Qwen3-TTS."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def transcribe(args):
    executable = Path(args.executable).resolve()
    if not executable.is_file():
        raise ValueError(f"ASR executable does not exist: {executable}")
    cmd = [str(executable), "transcribe", "--model", str(args.model),
           "--audio", str(args.audio), "--language", args.language, "--json"]
    return subprocess.run(cmd, check=False).returncode


def speak(args):
    if not args.model.is_dir():
        raise ValueError("--model must point to a downloaded local model directory")
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    # Never download a model in the inference command.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from mlx_audio.tts.utils import load_model
    except ImportError as e:
        raise ValueError("TTS requires the optional mlx-audio dependencies; see README") from e
    with contextlib.redirect_stdout(sys.stderr):
        model = load_model(args.model, strict=True)
        outputs = []
        rate = None
        for result in model.generate_custom_voice(text=args.text, speaker=args.speaker,
                                                   language=args.language):
            mx.eval(result.audio)
            current_rate = int(result.sample_rate)
            if rate is not None and rate != current_rate:
                raise ValueError("model returned inconsistent sample rates")
            rate = current_rate
            outputs.append(np.asarray(result.audio, dtype=np.float32))
        if not outputs or not rate:
            raise ValueError("model returned no audio")
        samples = np.concatenate(outputs)
        if samples.ndim != 1 or not np.isfinite(samples).all() or samples.size == 0:
            raise ValueError("model returned invalid audio")
        # Exclusive creation avoids silently overwriting a caller's file.
        with args.output.open("xb") as output:
            sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
    print(json.dumps({"status": "ok", "output": str(args.output),
                      "sample_rate": rate, "samples": int(samples.size),
                      "seconds": samples.size / rate}))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run supported voice models locally")
    subs = parser.add_subparsers(dest="task", required=True)
    p = subs.add_parser("transcribe", help="Qwen3-ASR BF16 native C/MLX backend")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--language", default="auto")
    p.add_argument("--executable", type=Path, default=ROOT / "build/qwen3-asr")
    p.set_defaults(handler=transcribe)
    p = subs.add_parser("speak", help="Qwen3-TTS CustomVoice via optional MLX Audio")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--speaker", default="Vivian")
    p.add_argument("--language", default="English")
    p.set_defaults(handler=speak)
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"voice: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
