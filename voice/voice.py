#!/usr/bin/env python3
"""Local voice CLI: native Qwen ASR and optional MLX Audio Qwen3-TTS."""
import argparse
import contextlib
import json
import inspect
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def mlx_audio_transcribe(model_path, paths, language):
    if not model_path.is_dir():
        raise ValueError("--model must point to a downloaded local model directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import mlx.core as mx
        from mlx_audio.stt.utils import load_model
    except ImportError as e:
        raise ValueError("MLX Audio requires optional dependencies; see README") from e
    with contextlib.redirect_stdout(sys.stderr):
        model = load_model(model_path.resolve())
        kwargs = {}
        if language != "auto":
            parameters = inspect.signature(model.generate).parameters
            if "language" not in parameters:
                raise ValueError("this model does not expose a language argument")
            kwargs["language"] = language
        records = []
        for path in paths:
            result = model.generate(str(path), **kwargs)
            mx.synchronize()
            if not isinstance(getattr(result, "text", None), str):
                raise ValueError("this model does not return a single text transcription")
            records.append({"audio": str(path), "text": result.text,
                            "status": "ok", "termination": "generate_returned"})
    return records


def transcribe_many(args):
    print(json.dumps({"results": mlx_audio_transcribe(args.model, args.audio, args.language)}, ensure_ascii=False))
    return 0

def transcribe(args):
    if args.backend == "mlx-audio":
        print(json.dumps(mlx_audio_transcribe(args.model, [args.audio], args.language)[0], ensure_ascii=False))
        return 0
    if args.backend == "parakeet":
        if not args.model.is_dir():
            raise ValueError("--model must point to a downloaded local model directory")
        if args.language not in ("auto", "English"):
            raise ValueError("Parakeet adapter currently supports auto or English only")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import load_audio, get_logmel
        except ImportError as e:
            raise ValueError("Parakeet requires optional dependencies; see README") from e
        with contextlib.redirect_stdout(sys.stderr):
            model = from_pretrained(str(args.model.resolve()))
            audio = load_audio(str(args.audio), model.preprocessor_config.sample_rate, mx.bfloat16)
            mx.eval(audio)
            result = model.generate(get_logmel(audio, model.preprocessor_config))[0]
            mx.synchronize()
        print(json.dumps({"text": result.text, "status": "ok", "termination": "generate_returned"}, ensure_ascii=False))
        return 0
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
    if args.backend == "kokoro":
        if (args.language or "a") not in ("a", "b"):
            raise ValueError("Kokoro adapter currently supports English language codes a or b")
        try:
            import spacy.util
        except ImportError as e:
            raise ValueError("Kokoro requires misaki[en] and the spaCy English model; see README") from e
        if not spacy.util.is_package("en_core_web_sm"):
            raise ValueError("Kokoro requires the local spaCy en_core_web_sm model; see README")
        # The tested espeakng-loader wheel points to its build-machine data path
        # on macOS; use the installed Homebrew library and data explicitly.
        try:
            import espeakng_loader
        except ImportError as e:
            raise ValueError("Kokoro requires espeakng-loader; see README") from e
        espeak_prefix = Path("/opt/homebrew/opt/espeak-ng")
        espeak_lib = espeak_prefix / "lib/libespeak-ng.dylib"
        espeak_data = espeak_prefix / "share/espeak-ng-data"
        if not espeak_lib.is_file() or not (espeak_data / "phontab").is_file():
            raise ValueError("Kokoro requires Homebrew espeak-ng; see README")
        espeakng_loader.get_library_path = lambda: str(espeak_lib)
        espeakng_loader.get_data_path = lambda: str(espeak_data)
    with contextlib.redirect_stdout(sys.stderr):
        model = load_model(args.model, strict=True)
        outputs = []
        rate = None
        if args.backend == "kokoro":
            results = model.generate(text=args.text, voice=args.speaker or "af_heart",
                                     lang_code=args.language or "a")
        else:
            results = model.generate_custom_voice(text=args.text, speaker=args.speaker or "Vivian",
                                                   language=args.language or "English")
        for result in results:
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
    p.add_argument("--backend", choices=("qwen", "parakeet", "mlx-audio"), default="qwen")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--language", default="auto")
    p.add_argument("--executable", type=Path, default=ROOT / "build/qwen3-asr")
    p.set_defaults(handler=transcribe)
    p = subs.add_parser("transcribe-many", help="Reuse one MLX Audio ASR model for sequential files")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--audio", type=Path, nargs="+", required=True)
    p.add_argument("--language", default="auto")
    p.set_defaults(handler=transcribe_many)
    p = subs.add_parser("speak", help="Qwen3-TTS or Kokoro through optional MLX Audio")
    p.add_argument("--backend", choices=("qwen", "kokoro"), default="qwen")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--speaker", default=None)
    p.add_argument("--language", default=None)
    p.set_defaults(handler=speak)
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"voice: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
