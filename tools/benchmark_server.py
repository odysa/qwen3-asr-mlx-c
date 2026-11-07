#!/usr/bin/env python3
"""Benchmark an already-running loopback ASR server; saves every request.

No models or datasets are downloaded. Input: frozen cohort JSON with rows having
case_id/config/audio_path/audio_sha256. Output directory must not already exist.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import urllib.error
import urllib.request
import uuid


def percentile(values, fraction):
    if not values: return None
    values = sorted(values); pos = (len(values)-1)*fraction
    lo = int(pos); hi = min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def upload(url, row, data, language):
    boundary = 'asr-'+uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\n{language}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
    request = urllib.request.Request(url+'/v1/audio/transcriptions',body,{'Content-Type':f'multipart/form-data; boundary={boundary}'})
    before = time.perf_counter()
    try:
        with urllib.request.urlopen(request,timeout=180) as response:
            result = json.loads(response.read())
            code = response.status
    except urllib.error.HTTPError as response:
        with response:
            raw = response.read().decode('utf-8', errors='replace')
            try:
                result = json.loads(raw)
                if not isinstance(result, dict):
                    result = {'error': raw}
            except json.JSONDecodeError:
                result = {'error': raw or 'empty HTTP error response'}
            result['raw_error_body'] = raw
            code = response.code
    except Exception as exc:
        result = {'error': type(exc).__name__+': '+str(exc)}; code = 0
    elapsed = time.perf_counter()-before
    return {'case_id':row['case_id'],'config':row['config'],'http_status':code,'client_s':elapsed,'response':result}


def get(url,path):
    with urllib.request.urlopen(url+path,timeout=10) as response: return json.loads(response.read())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--audio-root',type=Path,default=Path('.'))
    p.add_argument('--configs',nargs='+',default=['en_us','cmn_hans_cn'])
    p.add_argument('--languages-json',default='{"en_us":"en","cmn_hans_cn":"zh"}')
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--concurrency',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.url.startswith('http://127.0.0.1:'): p.error('URL must use http://127.0.0.1:PORT')
    if args.repeat < 1 or not 1 <= args.concurrency <= 32:p.error('invalid repeat/concurrency')
    args.output.mkdir(parents=True,exist_ok=False)
    cohort_bytes=args.cohort.read_bytes();cohort=json.loads(cohort_bytes)
    rows=[r for r in cohort['rows'] if r['config'] in args.configs]
    if not rows:raise ValueError('empty cohort')
    languages=json.loads(args.languages_json)
    audio={}
    for row in rows:
        data=(args.audio_root/row['audio_path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['audio_sha256']:raise ValueError('audio hash drift: '+row['case_id'])
        audio[row['case_id']]=data
    metadata={'protocol':'one warmup per selected language, then repeat-major requests; no retries',
              'platform':platform.platform(),'python':platform.python_version(),
              'cohort_sha256':hashlib.sha256(cohort_bytes).hexdigest(),
              'concurrency':args.concurrency,'repeat':args.repeat,'configs':args.configs,
              'languages':languages,'health':get(args.url,'/health'),'metrics_before':get(args.url,'/metrics'),
              'selected_inputs':[{k:r[k] for k in ('case_id','config','audio_path','audio_sha256','duration_s')} for r in rows]}
    def save(name,value): (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    save('PLAN.json',metadata)
    warmups=[]
    for config in args.configs:
        row=next((r for r in rows if r['config']==config),None)
        if row:warmups.append(upload(args.url,row,audio[row['case_id']],languages[config]))
    save('WARMUPS.json',warmups)
    jobs=[(repeat,r) for repeat in range(args.repeat) for r in rows]
    def run(job):
        repeat,row=job
        return dict(upload(args.url,row,audio[row['case_id']],languages[row['config']]),repeat=repeat)
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results=list(pool.map(run,jobs))
    wall=time.perf_counter()-start
    save('REQUESTS.json',results)
    metrics=get(args.url,'/metrics');save('METRICS.json',metrics)
    groups={}
    for config in args.configs:
        selected=[r for r in results if r['config']==config]
        ok=[r for r in selected if r['http_status']==200]
        groups[config]={'requests':len(selected),'completed':len(ok),'failed':len(selected)-len(ok),
            'client_p50_s':percentile([r['client_s'] for r in ok],.5),'client_p95_s':percentile([r['client_s'] for r in ok],.95),
            'inference_p50_s':percentile([r['response']['timing']['inference_s'] for r in ok],.5),
            'queue_p95_s':percentile([r['response']['timing']['queue_s'] for r in ok],.95)}
    successful=[r for r in results if r['http_status']==200]
    summary={'wall_s':wall,'requests':len(results),'failed':len(results)-len(successful),
             'requests_per_s':len(successful)/wall,
             'audio_seconds_per_wall_s':sum(r['response']['timing']['audio_s'] for r in successful)/wall,
             'groups':groups,'process_peak_rss_bytes':metrics['process_peak_rss_bytes'],
             'native_worker_peak_rss_bytes':metrics.get('native_worker_peak_rss_bytes')}
    save('SUMMARY.json',summary)
    # First repeat only. Explicit errors remain present, never silently removed.
    predictions=[]
    for r in results:
        if r['repeat']!=0:continue
        response=r['response'];ok=r['http_status']==200
        predictions.append({'case_id':r['case_id'],'status':'ok' if ok else 'error','text':response.get('text',''),
                            'termination_status':response.get('termination','error'),
                            'capped':response.get('possibly_capped',False)})
    for prediction, request in zip(predictions, [r for r in results if r['repeat']==0]):
        for key in ('eos', 'token_ids'):
            if key in request['response']:
                prediction[key] = request['response'][key]
    save('PREDICTIONS.json',predictions)
    pins={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output.iterdir() if f.is_file()}
    save('MANIFEST.json',pins)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
