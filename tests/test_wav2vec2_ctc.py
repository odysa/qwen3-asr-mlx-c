import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('wav2vec2_ctc', Path(__file__).resolve().parents[1] / 'voice/wav2vec2_ctc.py')
ctc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctc)
class CTCTest(unittest.TestCase):
    def test_repeated_ids_and_blank(self):
        self.assertEqual(ctc.decode_ctc([1,1,0,1,2,3,3], {0:'<pad>',1:'A',2:'|',3:'B'}, 0), 'AA B')
    def test_config_validation(self):
        cfg = {"model_type":"wav2vec2", "vocab_size":3, "hidden_size":4, "pad_token_id":0}
        proc = {"sampling_rate":16000, "do_normalize":True}
        tok = {"pad_token":"<pad>", "do_lower_case":False}
        vocab = {"<pad>":0, "|":1, "A":2}
        self.assertEqual(ctc.validate_config(cfg, proc, tok, vocab, (3,4), (3,))[2], "A")
        for patch in ({"do_lower_case":True}, {"word_delimiter_token":"#"}, {"pad_token":"[PAD]"}):
            with self.assertRaises(ValueError):
                ctc.validate_config(cfg, proc, {**tok, **patch}, vocab, (3,4), (3,))
        for cfg2, vocab2, shape, bias in [
            ({**cfg, "pad_token_id":1}, vocab, (3,4), (3,)),
            (cfg, {"<pad>":0, "|":1, "A":1}, (3,4), (3,)),
            (cfg, vocab, (2,4), (3,)), (cfg, vocab, (3,4), (2,))]:
            with self.assertRaises(ValueError):
                ctc.validate_config(cfg2, proc, tok, vocab2, shape, bias)

    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError): ctc.decode_ctc([3], {0:'<pad>'}, 0)
if __name__ == '__main__': unittest.main()
