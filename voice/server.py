"""Loopback-only, resident MLX Audio ASR service (Python 3.14).

One worker owns the model and serializes GPU access. HTTP admission is bounded;
this is request concurrency, not GPU batching or streaming recognition.
"""
import concurrent.futures
import contextlib
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import io
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import threading
import time
import uuid
import wave


class RequestError(ValueError):
    pass


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def rss_bytes():
    # ru_maxrss is bytes on macOS, KiB on Linux. Process lifetime high-water mark.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def parse_upload(content_type, body, max_seconds):
    """Accept only uploaded PCM16 mono 16-kHz WAVs, never a server-side path."""
    if "\r" in content_type or "\n" in content_type:
        raise RequestError("invalid Content-Type")
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    if message.get_content_type() != "multipart/form-data" or not message.is_multipart():
        raise RequestError("use multipart/form-data with a file field")
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name not in ("file", "model", "language", "response_format") or name in fields:
            raise RequestError("unknown or duplicate multipart field")
        value = part.get_payload(decode=True)
        if value is None or part.defects:
            raise RequestError("invalid multipart field")
        if name == "file":
            fields[name] = value
        else:
            if len(value) > 256:
                raise RequestError("text field too long")
            try:
                fields[name] = value.decode("utf-8")
            except UnicodeError as exc:
                raise RequestError("text fields must be UTF-8") from exc
    if not fields.get("file"):
        raise RequestError("file is required")
    if fields.get("response_format", "json") not in ("json", "text"):
        raise RequestError("response_format must be json or text")
    try:
        with wave.open(io.BytesIO(fields["file"]), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
                raise RequestError("audio must be uncompressed mono PCM16 WAV at 16000 Hz")
            frames = audio.getnframes()
            seconds = frames / 16000
            if not 0 < seconds <= max_seconds:
                raise RequestError(f"audio must be nonempty and at most {max_seconds:g} seconds")
            if len(audio.readframes(frames)) != frames * 2:
                raise RequestError("truncated WAV data")
    except (wave.Error, EOFError) as exc:
        raise RequestError("invalid WAV file") from exc
    return fields, seconds


class MLXAudioEngine:
    def __init__(self, model_path, options):
        if not model_path.is_dir():
            raise ValueError("--model must be an existing local model directory")
        if set(options) & {"audio", "language", "stream"}:
            raise ValueError("options cannot override audio, language or stream")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import mlx.core as mx
        from mlx_audio.stt.utils import load_model
        self.mx = mx
        self.options = options
        with contextlib.redirect_stdout(sys.stderr):
            self.model = load_model(model_path.resolve())
            # Materialize weights so first request does not pay lazy weight loading.
            mx.eval(self.model.parameters())
            mx.synchronize()
        self.parameters = inspect.signature(self.model.generate).parameters

    def transcribe(self, path, language):
        kwargs = dict(self.options)
        if language != "auto":
            if "language" not in self.parameters:
                raise RequestError("this model does not expose a language argument")
            kwargs["language"] = language
        with contextlib.redirect_stdout(sys.stderr):
            result = self.model.generate(str(path), **kwargs)
            self.mx.synchronize()
            text = getattr(result, "text", None)
            if not isinstance(text, str):
                raise RequestError("model does not return a single text result")
        record = {"text": text, "termination": "generate_returned"}
        count = getattr(result, "generation_tokens", None)
        if isinstance(count, int):
            record["generation_tokens"] = count
            limit = kwargs.get("max_tokens")
            if isinstance(limit, int) and limit > 0:
                record["possibly_capped"] = count >= limit
        return record


class ResidentService:
    def __init__(self, model_id, factory, max_pending=4):
        self.model_id = model_id
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-gpu")
        self.slots = threading.BoundedSemaphore(max_pending + 1)
        self.lock = threading.Lock()
        self.pending = 0
        self.accepted = self.completed = self.failed = self.rejected = 0
        self.latencies = []
        self.queues = []
        self.audio_seconds = 0.0
        self.started = time.perf_counter()
        def load():
            before = time.perf_counter()
            self.engine = factory()
            self.load_seconds = time.perf_counter() - before
        try:
            self.executor.submit(load).result()
        except BaseException:
            self.executor.shutdown(wait=True)
            raise
        self.ready = time.perf_counter()

    def submit(self, audio, seconds, language):
        if not self.slots.acquire(blocking=False):
            with self.lock:
                self.rejected += 1
            return None
        admitted = time.perf_counter()
        with self.lock:
            self.accepted += 1
            self.pending += 1
        def run():
            began = time.perf_counter()
            try:
                with tempfile.TemporaryDirectory(prefix="local-asr-") as folder:
                    path = Path(folder) / "audio.wav"
                    path.write_bytes(audio)
                    infer_start = time.perf_counter()
                    result = self.engine.transcribe(path, language)
                    inference_s = time.perf_counter() - infer_start
                end = time.perf_counter()
                timing = {"queue_s": began - admitted, "inference_s": inference_s,
                          "service_s": end - admitted, "audio_s": seconds,
                          "real_time_factor": inference_s / seconds}
                with self.lock:
                    self.completed += 1
                    self.audio_seconds += seconds
                    # Bound in-memory history; aggregate counters remain lifetime.
                    self.latencies.append(timing["service_s"])
                    self.queues.append(timing["queue_s"])
                    self.latencies = self.latencies[-10000:]
                    self.queues = self.queues[-10000:]
                return dict(result, model=self.model_id, timing=timing)
            except BaseException:
                with self.lock:
                    self.failed += 1
                raise
            finally:
                with self.lock:
                    self.pending -= 1
                self.slots.release()
        try:
            return self.executor.submit(run)
        except BaseException:
            with self.lock:
                self.pending -= 1
                self.failed += 1
            self.slots.release()
            raise

    def metrics(self):
        with self.lock:
            return {"model": self.model_id, "load_prepare_s": self.load_seconds,
                    "uptime_s": time.perf_counter() - self.ready,
                    "accepted": self.accepted, "completed": self.completed,
                    "failed": self.failed, "rejected_overload": self.rejected,
                    "pending_including_active": self.pending,
                    "service_latency_p50_s": percentile(self.latencies, .5),
                    "service_latency_p95_s": percentile(self.latencies, .95),
                    "queue_p95_s": percentile(self.queues, .95),
                    "latency_window_requests": len(self.latencies),
                    "completed_audio_s": self.audio_seconds,
                    "process_peak_rss_bytes": rss_bytes()}

    def close(self):
        self.executor.shutdown(wait=True)


class ASRServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, service, max_upload_bytes=16 * 1024 * 1024, max_audio_seconds=30):
        self.service = service
        self.max_upload_bytes = max_upload_bytes
        self.max_audio_seconds = max_audio_seconds
        super().__init__(address, ASRHandler)


class ASRHandler(BaseHTTPRequestHandler):
    server_version = "LocalASR/1"
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *_):
        pass

    def respond(self, status, value, content_type="application/json"):
        body = json.dumps(value, ensure_ascii=False).encode() if content_type == "application/json" else value.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"status": "ready", "model": self.server.service.model_id})
        elif self.path == "/metrics":
            self.respond(200, self.server.service.metrics())
        elif self.path == "/v1/models":
            self.respond(200, {"object": "list", "data": [{"id": self.server.service.model_id, "object": "model"}]})
        else:
            self.respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/audio/transcriptions":
            self.respond(404, {"error": "not found"})
            return
        try:
            if self.headers.get("Origin"):
                raise RequestError("browser-origin requests are unsupported")
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
                raise RequestError("one Content-Length is required; chunked uploads are unsupported")
            try:
                size = int(lengths[0])
            except ValueError as exc:
                raise RequestError("invalid Content-Length") from exc
            if not 0 < size <= self.server.max_upload_bytes:
                self.respond(413, {"error": "upload exceeds byte limit or is empty"})
                return
            body = self.rfile.read(size)
            if len(body) != size:
                raise RequestError("incomplete upload")
            fields, seconds = parse_upload(self.headers.get("Content-Type", ""), body, self.server.max_audio_seconds)
            if fields.get("model", self.server.service.model_id) != self.server.service.model_id:
                raise RequestError("requested model is not loaded")
            language = fields.get("language", "auto")
            future = self.server.service.submit(fields["file"], seconds, language)
            if future is None:
                self.respond(429, {"error": "model queue full; retry later"})
                return
            result = future.result()
            result["id"] = uuid.uuid4().hex
            if fields.get("response_format") == "text":
                self.respond(200, result["text"], "text/plain")
            else:
                self.respond(200, result)
        except (RequestError, UnicodeError) as exc:
            self.respond(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            # An admitted GPU job is allowed to finish; its result remains counted.
            pass
        except Exception:
            self.respond(500, {"error": "transcription failed; inspect server stderr"})
            import traceback
            traceback.print_exc(file=sys.stderr)


def serve(args):
    if args.host != "127.0.0.1":
        raise ValueError("local server binds only to 127.0.0.1")
    if not 0 <= args.port <= 65535 or not 0 <= args.max_pending <= 64:
        raise ValueError("invalid port or max-pending")
    if not 0 < args.max_audio_seconds <= 3600 or not 0 < args.max_upload_mb <= 256:
        raise ValueError("invalid upload or audio limit")
    options = json.loads(args.options_json)
    if not isinstance(options, dict):
        raise ValueError("--options-json must be an object")
    service = ResidentService(args.model_id or args.model.name,
                              lambda: MLXAudioEngine(args.model, options), args.max_pending)
    try:
        with ASRServer((args.host, args.port), service, int(args.max_upload_mb * 1024 * 1024), args.max_audio_seconds) as server:
            print(json.dumps({"status": "ready", "url": f"http://{args.host}:{server.server_port}",
                              "model": service.model_id, "load_prepare_s": service.load_seconds}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        service.close()
    return 0
