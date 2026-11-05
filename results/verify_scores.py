#!/usr/bin/env python3
"""Recompute the published error counts from bundled predictions and labels."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
root = Path(__file__).resolve().parent
expected = {
 'competitors': {'qwen-bf16': (10,38), 'qwen-q8':(10,38), 'whisper-small':(15,55), 'whisper-turbo':(10,30), 'parakeet-v3':(10,None)},
 'q4': {'qwen-q8':(10,38), 'qwen-q4':(11,40), 'parakeet-v3':(10,None)},
 'head8': {'qwen-q8':(10,38), 'qwen-head8':(10,38), 'parakeet-v3':(10,None)},
}
with tempfile.TemporaryDirectory() as tmp:
 for group, models in expected.items():
  for name, (en, zh) in models.items():
   path = root/group
   output = Path(tmp) / 'result.json'
   subprocess.run([sys.executable, str(root/'quality/score_predictions.py'), '--implementation',str(path/(name+'.implementation.json')), '--predictions',str(path/(name+'.predictions.json')), '--output',str(output)],check=True,capture_output=True,text=True)
   scores=json.loads(output.read_text())['aggregates']
   assert scores['en_us']['errors']==en and scores['en_us']['reference_units']==212, (group,name,'en')
   if zh is not None:
    assert scores['cmn_hans_cn']['errors']==zh and scores['cmn_hans_cn']['reference_units']==367, (group,name,'zh')
   else:
    assert scores['cmn_hans_cn']['supported']==0, (group,name,'zh scope')
   print(group, name, 'OK')
