import json, resource, sys, time
import mlx.core as mx
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import load_audio, get_logmel
model_path, wav, warm_s, repeat_s = sys.argv[1:]
warm, repeat = int(warm_s), int(repeat_s)
if warm < 0 or repeat < 1: raise SystemExit(2)
t0=time.monotonic(); model=from_pretrained(model_path); mx.synchronize(); load_s=time.monotonic()-t0
audio=load_audio(wav, model.preprocessor_config.sample_rate, mx.bfloat16); mx.eval(audio); mx.synchronize()
print(json.dumps({'load_s':load_s}),flush=True)
for i in range(warm+repeat):
    t=time.monotonic(); mel=get_logmel(audio,model.preprocessor_config); result=model.generate(mel)[0]; text=result.text; mx.synchronize(); elapsed=time.monotonic()-t
    print(json.dumps({'run':i,'warmup':i<warm,'total_s':elapsed,'status':'ok','termination':'generate_returned','rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'text':text},ensure_ascii=False),flush=True)
