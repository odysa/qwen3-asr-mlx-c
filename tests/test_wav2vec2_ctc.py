import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('wav2vec2_ctc', Path(__file__).resolve().parents[1] / 'voice/wav2vec2_ctc.py')
ctc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctc)
class CTCTest(unittest.TestCase):
    def test_repeated_ids_and_blank(self):
        self.assertEqual(ctc.decode_ctc([1,1,0,1,2,3,3], {0:'<pad>',1:'A',2:'|',3:'B'}, 0), 'AA B')
    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError): ctc.decode_ctc([3], {0:'<pad>'}, 0)
if __name__ == '__main__': unittest.main()
