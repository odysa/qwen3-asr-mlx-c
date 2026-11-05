"""CPU-only transcript scoring; never imports or calls a model."""
import json,unicodedata,argparse
from pathlib import Path
NORMALIZATION_VERSION='nfkc-casefold-punct-symbol-space-v1'
def normalize(text,config):
 text=unicodedata.normalize('NFKC',text).casefold()
 text=''.join(' ' if unicodedata.category(c)[0] in 'PS' else c for c in text)
 text=' '.join(text.split())
 return text.split() if config=='en_us' else list(''.join(text.split()))
def edit_counts(ref,hyp):
 # Tuple=(distance,S,D,I); deterministic tie preference match/sub,delete,insert.
 row=[(i,0,0,i) for i in range(len(hyp)+1)]
 for i,a in enumerate(ref,1):
  nxt=[(i,0,i,0)]
  for j,b in enumerate(hyp,1):
   d,s,x,y=row[j-1];sub=(d+(a!=b),s+(a!=b),x,y)
   d,s,x,y=row[j];delete=(d+1,s,x+1,y)
   d,s,x,y=nxt[j-1];insert=(d+1,s,x,y+1)
   nxt.append(min((sub,delete,insert),key=lambda t:t[0]))
  row=nxt
 return row[-1]
def score(cohort,predictions):
 expected={r['case_id'] for r in cohort['rows']}
 if not isinstance(predictions,list):raise ValueError('predictions must be list')
 pred={}
 for p in predictions:
  if p['case_id'] in pred:raise ValueError('duplicate prediction')
  if p['case_id'] not in expected:raise ValueError('unknown case')
  pred[p['case_id']]=p
 results=[]
 for r in cohort['rows']:
  if not r['supported']:
   results.append(dict(case_id=r['case_id'],config=r['config'],status='unsupported',reason=r['reason']));continue
  p=pred.get(r['case_id']);failed=p is None or p.get('status')!='ok'
  if not failed and not isinstance(p.get('text'),str):raise ValueError('ok prediction must have string text')
  h='' if failed else p['text'];ref=normalize(r['transcription'],r['config']);hyp=normalize(h,r['config'])
  if not ref:raise ValueError('empty normalized reference')
  d,s,de,ins=edit_counts(ref,hyp)
  results.append(dict(case_id=r['case_id'],config=r['config'],status='missing_or_failed' if failed else 'ok',reference_units=len(ref),hypothesis_units=len(hyp),S=s,D=de,I=ins,errors=d,rate=d/len(ref),hypothesis=h,normalized_ref=ref,normalized_hyp=hyp))
 aggregates={}
 for lang in ['en_us','cmn_hans_cn']:
  selected=[r for r in results if r['config']==lang];supported=[r for r in selected if r['status']!='unsupported'];n=sum(r['reference_units'] for r in supported);e=sum(r['errors'] for r in supported)
  aggregates[lang]=dict(metric='WER' if lang=='en_us' else 'CER',selected=len(selected),supported=len(supported),unsupported=len(selected)-len(supported),completed=sum(r['status']=='ok' for r in supported),failed=sum(r['status']=='missing_or_failed' for r in supported),reference_units=n,errors=e,rate=e/n if n else None)
 return dict(normalization=NORMALIZATION_VERSION,failure_policy='failed/missing supported rows scored as empty hypothesis; never drop failures',aggregates=aggregates,rows=results)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--cohort',required=True);a.add_argument('--predictions',required=True);a.add_argument('--output',required=True);args=a.parse_args()
 Path(args.output).write_text(json.dumps(score(json.loads(Path(args.cohort).read_text()),json.loads(Path(args.predictions).read_text())),ensure_ascii=False,indent=2)+'\n')
