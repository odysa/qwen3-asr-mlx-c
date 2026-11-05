"""CPU-only wrapper around the byte-identical frozen Oct31 WER/CER scorer."""
import argparse,copy,hashlib,json,math
from pathlib import Path
import frozen_score
Q=Path(__file__).resolve().parent
COHORT_SHA='d91f418f2527c86df8da5182ca60afe18beac817d32d017174366d1b0d28c17a'
SCORER_SHA='851fbb1278be2893bce819923e25bcce98108c88c5c77ef200dbc63165a5f2ef'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_json(path):
 def unique(pairs):
  d={}
  for k,v in pairs:
   if k in d:raise ValueError('duplicate JSON key '+k)
   d[k]=v
  return d
 return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))
def evaluate(cohort,implementation,predictions):
 for key in ['implementation_id','model_id','precision']:
  if not isinstance(implementation.get(key),str) or not implementation[key].strip():raise ValueError('implementation manifest requires '+key)
 configs=implementation.get('evaluated_configs')
 if not isinstance(configs,list) or not configs or len(set(configs))!=len(configs) or not set(configs)<={'en_us','cmn_hans_cn'}:raise ValueError('invalid evaluated_configs')
 if len(configs)<2 and not implementation.get('scope_note'):raise ValueError('subset requires explicit scope_note')
 if not isinstance(predictions,list):raise ValueError('predictions must be JSON array')
 known={r['case_id']:r for r in cohort['rows']};seen=set();normalized=[];raw_rows=[];warnings=[]
 for p in predictions:
  if not isinstance(p,dict) or p.get('case_id') not in known:raise ValueError('unknown/missing case_id')
  cid=p['case_id']
  if cid in seen:raise ValueError('duplicate case_id '+cid)
  seen.add(cid)
  if p.get('status') not in ['ok','error','timeout','truncated','unsupported']:raise ValueError('invalid status '+cid)
  if 'text' in p and not isinstance(p['text'],str):raise ValueError('text must be string')
  for key in ['capped','truncated','eos']:
   if key in p and type(p[key]) is not bool:raise ValueError('invalid boolean '+key)
  status=p['status']
  if status=='ok':
   if 'text' not in p:raise ValueError('ok result requires raw text')
   if p.get('capped') is True or p.get('truncated') is True or p.get('eos') is False:status='truncated'
   if 'eos' not in p and 'termination_status' not in p:warnings.append(cid+': termination evidence not exposed; status is adapter-reported')
  if status=='unsupported' and known[cid]['config'] in configs:warnings.append(cid+': unplanned unsupported within evaluated scope counts as failure')
  normalized.append(dict(case_id=cid,status=status,text=p.get('text','')))
  raw_rows.append(copy.deepcopy(p))
 c=copy.deepcopy(cohort)
 for r in c['rows']:
  if r['config'] not in configs:r.update(supported=False,reason='not evaluated under predeclared implementation scope: '+implementation['scope_note'])
 result=frozen_score.score(c,normalized)
 result.update(implementation=copy.deepcopy(implementation),selected_cases=len(c['rows']),received_prediction_rows=len(predictions),missing_case_ids=sorted(set(known)-seen),raw_prediction_rows=raw_rows,warnings=warnings,scope='per-language descriptive diagnostic; no cross-language aggregate, causal inference or general quality-equivalence claim')
 return result
def main():
 a=argparse.ArgumentParser();a.add_argument('--implementation',type=Path,required=True);a.add_argument('--predictions',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 if sha(Q/'COHORT.json')!=COHORT_SHA or sha(Q/'frozen_score.py')!=SCORER_SHA:raise ValueError('frozen input/scorer drift')
 paths=[Q/'COHORT.json',Q/'frozen_score.py',args.implementation,args.predictions];pins={str(p.resolve()):sha(p) for p in paths}
 if args.output.resolve() in {p.resolve() for p in paths}:raise ValueError('output overlaps input')
 result=evaluate(load_json(Q/'COHORT.json'),load_json(args.implementation),load_json(args.predictions));result['input_sha256']=pins
 if any(sha(Path(p))!=h for p,h in pins.items()):raise ValueError('input changed while scoring')
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result['aggregates'],indent=2))
if __name__=='__main__':main()
