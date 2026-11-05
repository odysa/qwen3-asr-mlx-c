"""Interface checks without loading models or running GPU inference."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('local_voice', Path(__file__).resolve().parents[1] / 'voice/voice.py')
voice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voice)

class GenericASRTest(unittest.TestCase):
    def test_batch_reuses_model_and_passes_language(self):
        loaded, calls = [], []
        class Model:
            def generate(self, path, language=None):
                calls.append((path, language))
                return types.SimpleNamespace(text='hello')
        def load(path):
            loaded.append(path)
            return Model()
        mlx = types.ModuleType('mlx'); core = types.ModuleType('mlx.core')
        core.synchronize = lambda: None
        mlx.core = core
        audio = types.ModuleType('mlx_audio'); audio.__path__ = []
        stt = types.ModuleType('mlx_audio.stt'); stt.__path__ = []
        utils = types.ModuleType('mlx_audio.stt.utils'); utils.load_model = load
        modules = {'mlx': mlx, 'mlx.core': core, 'mlx_audio': audio,
                   'mlx_audio.stt': stt, 'mlx_audio.stt.utils': utils}
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, modules):
            results = voice.mlx_audio_transcribe(Path(folder), [Path('a.wav'), Path('b.wav')], 'en')
            self.assertEqual(loaded, [Path(folder).resolve()])
            self.assertEqual(calls, [('a.wav', 'en'), ('b.wav', 'en')])
            self.assertEqual([item['text'] for item in results], ['hello', 'hello'])

    def test_options_must_be_json_object(self):
        self.assertEqual(voice.parse_options('{"temperature": 0}'), {"temperature": 0})
        for value in ('[]', 'not json'):
            with self.assertRaises(ValueError):
                voice.parse_options(value)

    def test_missing_model_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'local model'):
                voice.mlx_audio_transcribe(Path(folder) / 'missing', [Path('a.wav')], 'auto')

if __name__ == '__main__':
    unittest.main()
