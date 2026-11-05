"""Download pinned public model and official example WAVs; no inference."""
import argparse, concurrent.futures, datetime, hashlib, json, urllib.request, wave
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REVISIONS={"0.6B":"5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
           "1.7B":"7278e1e70fe206f11671096ffdd38061171dd6e5"}
OFFICIAL_REV="7c6daf77a2421100f5fb066495372c00129d39ff"
def fetch(item):
    url,path=item;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_suffix(path.suffix+".part")
        with urllib.request.urlopen(url,timeout=120) as src, tmp.open("wb") as dst:
            while block:=src.read(1024*1024):dst.write(block)
        tmp.rename(path)
    h=hashlib.sha256()
    with path.open("rb") as f:
        while b:=f.read(1024*1024):h.update(b)
    return {"url":url,"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":h.hexdigest()}
def main():
    parser=argparse.ArgumentParser(description="Download a pinned Qwen3-ASR model and example WAVs")
    parser.add_argument("--size", choices=sorted(REVISIONS), default="0.6B")
    size=parser.parse_args().size
    MODEL=f"Qwen/Qwen3-ASR-{size}"; REV=REVISIONS[size]
    api=json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{MODEL}/revision/{REV}"))
    dest=ROOT/"models"/f"Qwen3-ASR-{size}"
    names=[f["rfilename"] for f in api["siblings"] if f["rfilename"]!=".gitattributes"]
    work=[(f"https://huggingface.co/{MODEL}/resolve/{REV}/{n}",dest/n) for n in names]
    work += [(f"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_{lang}.wav", ROOT/"fixtures"/f"asr_{lang}.wav") for lang in ["en","zh"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,work))
    audio=[]
    for lang in ["en","zh"]:
        with wave.open(str(ROOT/"fixtures"/f"asr_{lang}.wav")) as w:audio.append({"id":lang,"path":f"fixtures/asr_{lang}.wav","channels":w.getnchannels(),"sample_rate":w.getframerate(),"frames":w.getnframes(),"sample_width":w.getsampwidth(),"seconds":w.getnframes()/w.getframerate(),"ground_truth":None,"quality_status":"no independently pinned transcript; output agreement only"})
    result={"captured_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"model":MODEL,"revision":REV,"official_audio_readme":f"https://github.com/QwenLM/Qwen3-ASR/blob/{OFFICIAL_REV}/README.md","files":records,"audio":audio,"inference_calls":0}
    (ROOT/"benchmarks"/("INPUT_MANIFEST.json" if size == "0.6B" else f"INPUT_MANIFEST_{size}.json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"revision":REV,"files":len(records),"audio":audio},indent=2),flush=True)
if __name__=="__main__":main()
