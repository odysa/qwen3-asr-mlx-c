"""Real HTTP/lifecycle checks with a deterministic engine, no MLX import required."""
import concurrent.futures
import io
import importlib.util
import json
from pathlib import Path
import sys
import threading
import tempfile
import socket
import time
import unittest
import urllib.error
import urllib.request
import wave
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'voice'))
from server import ASRServer, ResidentService, RequestError, parse_upload, NativeEngine


def wav_bytes(rate=16000):
    output = io.BytesIO()
    with wave.open(output, 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes(b'\x00\x00' * 160)
    return output.getvalue()


def multipart(audio, extra=None):
    body = b'--test\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n' + audio + b'\r\n'
    for key, value in (extra or {}).items():
        body += f'--test\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
    return body + b'--test--\r\n'


class ServerTests(unittest.TestCase):
    def test_wav_validation(self):
        with self.assertRaises(RequestError):
            parse_upload('multipart/form-data; boundary=test', multipart(wav_bytes(8000)), 30)
        with self.assertRaises(RequestError):
            parse_upload('multipart/form-data; boundary=test', multipart(wav_bytes()[:-20]), 30)
        with self.assertRaises(RequestError):
            parse_upload('multipart/form-data; boundary=test', multipart(wav_bytes(), {'path': '/etc/passwd'}), 30)
        with self.assertRaises(RequestError):
            parse_upload('multipart/form-data; boundary=test', multipart(wav_bytes()), .001)

    def test_single_owner_load_once_bounded_queue_and_error_recovery(self):
        entered, release = threading.Event(), threading.Event()
        threads = []
        class Engine:
            def __init__(self): threads.append(threading.get_ident())
            def transcribe(self, path, language):
                threads.append(threading.get_ident())
                if language == 'hold': entered.set(); release.wait(3)
                if language == 'fail': raise RuntimeError('failure')
                return {'text': 'hello', 'termination': 'generate_returned'}
        service = ResidentService('fake', Engine, max_pending=0)
        try:
            first = service.submit(wav_bytes(), .01, 'hold')
            self.assertTrue(entered.wait(2))
            self.assertIsNone(service.submit(wav_bytes(), .01, 'auto'))
            release.set(); self.assertEqual(first.result(2)['text'], 'hello')
            failed = service.submit(wav_bytes(), .01, 'fail')
            with self.assertRaises(RuntimeError): failed.result(2)
            self.assertEqual(service.submit(wav_bytes(), .01, 'auto').result(2)['text'], 'hello')
            self.assertEqual(len(set(threads)), 1)
            metrics = service.metrics()
            self.assertEqual((metrics['accepted'],metrics['completed'],metrics['failed'],metrics['rejected_overload'],metrics['pending_including_active']), (3,2,1,1,0))
        finally: release.set(); service.close()

    def test_native_pipe_reuses_worker_and_closes_it(self):
        script = """#!/usr/bin/env python3
import json,sys
print(json.dumps({'status':'ready'}),flush=True)
for line in sys.stdin:
    language,path=line.rstrip('\\n').split('\\t')
    print(json.dumps({'text':language,'eos':True,'native_peak_rss_bytes':123}),flush=True)
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'worker'
            path.write_text(script);path.chmod(0o755)
            engine = NativeEngine(Path(folder), path)
            try:
                pid = engine.process.pid
                self.assertEqual(engine.transcribe(Path('a.wav'), 'en')['text'], 'English')
                self.assertEqual(engine.transcribe(Path('b.wav'), 'zh')['text'], 'Chinese')
                self.assertEqual(engine.process.pid, pid)
                self.assertEqual(engine.peak_rss_bytes, 123)
                with self.assertRaises(RequestError): engine.transcribe(Path('a.wav'), 'fr')
                self.assertTrue(engine.is_ready())
            finally: engine.close()
            self.assertFalse(engine.is_ready())
            self.assertIsNotNone(engine.process.poll())

    def test_native_partial_line_has_full_response_deadline(self):
        script = """#!/usr/bin/env python3
import json,sys,time
print(json.dumps({'status':'ready'}),flush=True)
for line in sys.stdin:
    print('{"text":',end='',flush=True)
    time.sleep(30)
"""
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'worker';path.write_text(script);path.chmod(0o755)
            engine=NativeEngine(Path(folder),path,response_timeout_s=3)
            engine.response_timeout_s=.2
            before=time.monotonic()
            try:
                with self.assertRaisesRegex(RuntimeError,'deadline'):
                    engine.transcribe(Path('a.wav'),'en')
                self.assertLess(time.monotonic()-before,2)
                self.assertFalse(engine.is_ready())
                self.assertIsNotNone(engine.process.poll())
            finally:engine.close()

    def test_native_invalid_protocol_marks_worker_failed(self):
        for response in ('not json', '[]', '{}', '{"text":"partial","eos":false}'):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as folder:
                script = "#!/usr/bin/env python3\nimport sys,json\nprint(json.dumps({'status':'ready'}),flush=True)\nfor line in sys.stdin: print("+repr(response)+",flush=True)\n"
                path=Path(folder)/'worker';path.write_text(script);path.chmod(0o755)
                engine=NativeEngine(Path(folder),path,response_timeout_s=3)
                try:
                    with self.assertRaises(RuntimeError):engine.transcribe(Path('a.wav'),'en')
                    self.assertFalse(engine.is_ready())
                    self.assertIsNotNone(engine.process.poll())
                finally:engine.close()

    def test_benchmark_retains_empty_and_nonjson_http_errors(self):
        spec = importlib.util.spec_from_file_location('benchmark_server', Path(__file__).resolve().parents[1] / 'tools/benchmark_server.py')
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for body in (b'', b'<html>overloaded</html>'):
            error = urllib.error.HTTPError('http://127.0.0.1:1',503,'overloaded',{},io.BytesIO(body))
            with patch('urllib.request.urlopen',side_effect=error):
                record = module.upload('http://127.0.0.1:1',{'case_id':'case','config':'en_us'},wav_bytes(),'en')
            self.assertEqual(record['http_status'],503)
            self.assertEqual(record['response']['raw_error_body'],body.decode())
            self.assertIn('error',record['response'])

    def test_connection_admission_limits_unfinished_uploads(self):
        service = ResidentService('fake', lambda: object())
        server = ASRServer(('127.0.0.1',0),service,max_connections=2)
        thread = threading.Thread(target=server.serve_forever);thread.start()
        sockets=[]
        try:
            # Leave two bodies unfinished; neither may reach the GPU queue.
            for _ in range(2):
                sock=socket.create_connection(server.server_address,timeout=2)
                sock.sendall(b'POST /v1/audio/transcriptions HTTP/1.0\r\nContent-Length: 100\r\n\r\n')
                sockets.append(sock)
            third=socket.create_connection(server.server_address,timeout=2)
            sockets.append(third)
            self.assertIn(b'503 Service Unavailable',third.recv(1024))
            self.assertEqual(service.metrics()['accepted'],0)
        finally:
            for sock in sockets:sock.close()
            server.shutdown();thread.join();server.server_close();service.close()

    def test_http_upload_models_metrics_and_text(self):
        class Engine:
            def transcribe(self, path, language):
                self.last_path = path
                return {'text': '你好', 'termination': 'generate_returned'}
        engine = Engine(); service = ResidentService('fake', lambda: engine)
        server = ASRServer(('127.0.0.1',0),service,max_upload_bytes=1024)
        thread = threading.Thread(target=server.serve_forever);thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        def request(path, body=None, headers=None):
            req = urllib.request.Request(url+path,data=body,headers=headers or {})
            try:
                with urllib.request.urlopen(req,timeout=3) as r:return r.status,r.read()
            except urllib.error.HTTPError as r:
                with r:return r.code,r.read()
        try:
            self.assertEqual(request('/health')[0],200)
            self.assertEqual(json.loads(request('/v1/models')[1])['data'][0]['id'],'fake')
            status,body=request('/v1/audio/transcriptions',multipart(wav_bytes()),{'Content-Type':'multipart/form-data; boundary=test'})
            self.assertEqual(status,200);self.assertEqual(json.loads(body)['text'],'你好')
            self.assertFalse(engine.last_path.exists())
            status,body=request('/v1/audio/transcriptions',multipart(wav_bytes(),{'response_format':'text'}),{'Content-Type':'multipart/form-data; boundary=test'})
            self.assertEqual((status,body),(200,'你好'.encode()))
            self.assertEqual(request('/v1/audio/transcriptions',multipart(wav_bytes(),{'model':'other'}),{'Content-Type':'multipart/form-data; boundary=test'})[0],400)
            self.assertEqual(request('/v1/audio/transcriptions',b'x'*1025,{'Content-Type':'audio/wav'})[0],413)
            self.assertEqual(request('/unknown')[0],404)
            metrics=json.loads(request('/metrics')[1]);self.assertEqual(metrics['completed'],2)
            engine.is_ready=lambda:False
            self.assertEqual(request('/health')[0],503)
        finally:
            server.shutdown();thread.join();server.server_close();service.close()


if __name__ == '__main__': unittest.main()
