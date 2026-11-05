"""Normalize official WAV inputs reproducibly; retain original files."""
import hashlib,json,math,wave
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1]
def main():
    records=[]
    for lang in ["en","zh"]:
        source=ROOT/"fixtures"/f"asr_{lang}.wav";sr,x=wavfile.read(source)
        if x.ndim!=1:raise ValueError("Expected mono")
        if np.issubdtype(x.dtype,np.integer):x=x.astype(np.float64)/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
        if sr!=16000:
            g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g,window=("kaiser",5.0),padtype="constant")
        pcm=np.clip(np.rint(x*32768),-32768,32767).astype(np.int16)
        dest=ROOT/"fixtures"/f"asr_{lang}_16k_pcm16.wav";wavfile.write(dest,16000,pcm)
        records.append({"id":lang,"source":str(source.relative_to(ROOT)),"path":str(dest.relative_to(ROOT)),"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"frames":len(pcm),"seconds":len(pcm)/16000,"language":{"en":"English","zh":"Chinese"}[lang],"reference_text":None,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest()})
    (ROOT/"benchmarks"/"FIXTURES.json").write_text(json.dumps({"method":"scipy resample_poly Kaiser5 constant pad; float64; rint(x*32768) clipped int16; original retained","cases":records},indent=2)+"\n")
    print(json.dumps(records,indent=2))
if __name__=="__main__":main()
