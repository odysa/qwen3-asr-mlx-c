#!/usr/bin/env python3
"""Recompute serving quality and latency tables from retained HTTP requests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'results/quality'))
from score_predictions import evaluate
from benchmark_server import percentile


def summarize(folder):
    cohort=json.loads((ROOT/'results/quality/COHORT.json').read_text())
    output={}
    for path in sorted(folder.glob('*-c[14]')):
        if not path.is_dir():continue
        plan=json.loads((path/'PLAN.json').read_text())
        records=json.loads((path/'REQUESTS.json').read_text())
        predictions=json.loads((path/'PREDICTIONS.json').read_text())
        summary=json.loads((path/'SUMMARY.json').read_text())
        metrics=json.loads((path/'METRICS.json').read_text())
        model=plan['health']['model'];concurrency=plan['concurrency']
        manifest=json.loads((path/'MANIFEST.json').read_text())
        for name,h in manifest.items():
            if hashlib.sha256((path/name).read_bytes()).hexdigest()!=h:raise ValueError('hash mismatch: '+str(path/name))
        expected={(row['case_id'],repeat) for row in plan['selected_inputs'] for repeat in range(plan['repeat'])}
        actual={(r['case_id'],r['repeat']) for r in records}
        if len(records)!=len(expected) or actual!=expected:raise ValueError('missing or duplicate request')
        first={r['case_id']:r for r in records if r['repeat']==0}
        for prediction in predictions:
            request=first[prediction['case_id']]
            if request['http_status']==200 and prediction['text']!=request['response']['text']:raise ValueError('first-repeat extraction changed')
        implementation={'implementation_id':model,'model_id':model,'precision':folder.joinpath('PROFILES.json').exists() and json.loads((folder/'PROFILES.json').read_text())[model]['precision'] or 'see profile description',
                        'evaluated_configs':plan['configs'],'scope_note':'Fixed supported-language cohort; other language not evaluated'}
        scores=evaluate(cohort,implementation,predictions)
        seen_texts={cid:{r['response'].get('text') for r in records if r['case_id']==cid and r['http_status']==200} for cid in first}
        repeated_changes=[cid for cid,values in seen_texts.items() if len(values)>1]
        success=[r for r in records if r['http_status']==200]
        if summary['requests']!=len(records) or summary['failed']!=len(records)-len(success):raise ValueError('failure count drift')
        if summary['requests_per_s']!=len(success)/summary['wall_s']:raise ValueError('throughput drift')
        if summary['audio_seconds_per_wall_s']!=sum(r['response']['timing']['audio_s'] for r in success)/summary['wall_s']:raise ValueError('audio throughput drift')
        for config,g in summary['groups'].items():
            ok=[r for r in records if r['config']==config and r['http_status']==200]
            for key,values in [('client_p50_s',[r['client_s'] for r in ok]),('client_p95_s',[r['client_s'] for r in ok]),('inference_p50_s',[r['response']['timing']['inference_s'] for r in ok]),('queue_p95_s',[r['response']['timing']['queue_s'] for r in ok])]:
                expected_value=percentile(values,.95 if 'p95' in key else .5)
                if g[key]!=expected_value:raise ValueError('summary drift: '+key)
        output[path.name]={'model':model,'concurrency':concurrency,'summary':summary,'quality':scores['aggregates'],
                           'later_repeat_text_changes':repeated_changes,'load_prepare_s':metrics['load_prepare_s'],
                           'load_boundary':metrics.get('load_boundary'),
                           'warmups':json.loads((path/'WARMUPS.json').read_text())}
    if not output:raise ValueError('no serving request bundles')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results/serving-v1')
    p.add_argument('--check',action='store_true')
    args=p.parse_args();result=summarize(args.results)
    if args.check:
        if json.loads((args.results/'SCORES_AND_METRICS.json').read_text())!=result:raise ValueError('published summary mismatch')
        print(f"Verified {len(result)} serving bundles, first-repeat scores and latency percentiles")
    else:
        (args.results/'SCORES_AND_METRICS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(f"Summarized {len(result)} serving bundles")

if __name__=='__main__':main()
