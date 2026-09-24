"""Supplement frozen summaries with predeclared seed-bootstrap movement CIs."""
from pathlib import Path
import argparse,csv,json,hashlib
from datetime import datetime,timezone
import numpy as np
from analysis_evidence_gate import validate

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def truth(x):return x=='True'

def summarize(base,out):
    validate(base,phase='science')
    summary=read(base/'analysis/summary.json');cfg=read(ROOT/'config/extension.json')
    if summary['phase']!='science' or summary['status']!='complete_verified_descriptive':raise ValueError('COMPLETE_SCIENCE_ONLY')
    seeds=cfg['seeds'] if summary['host']=='mta' else cfg['replication_seeds']
    if summary['observed_cells']!=8*len(seeds):raise ValueError('INCOMPLETE_CELLS')
    with (base/'analysis/paired_endpoints.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len(rows)!=32*len(seeds):raise ValueError('INCOMPLETE_PAIRED_GRID')
    indices=np.random.default_rng(cfg['bootstrap_seed']).integers(0,len(seeds),size=(cfg['bootstrap_replicates'],len(seeds)))
    results=[]
    for g in summary['groups']:
        rr={int(r['seed']):r for r in rows if r['ranker_id']==g['ranker_id'] and r['condition']==g['condition']}
        if set(rr)!=set(seeds):raise ValueError('SEED_GRID')
        eligible=np.array([truth(rr[s]['eligible']) and truth(rr[s]['nonempty']) for s in seeds])
        moved=np.array([truth(rr[s]['changed']) for s in seeds])
        movement=np.array([float(rr[s]['movement_fraction']) if rr[s]['movement_fraction'] else 0.0 for s in seeds])
        denom=eligible[indices].sum(1);undefined=int(np.sum(denom==0))
        record={'ranker_id':g['ranker_id'],'condition':g['condition'],'seeds':len(seeds),'eligible_nonempty':int(eligible.sum()),
                'bootstrap_replicates':len(indices),'undefined_denominator_resamples':undefined,
                'ci_status':'available' if not undefined else 'UNAVAILABLE_ZERO_BOOTSTRAP_DENOMINATOR',
                'composition_change_fraction':g['composition_change_fraction'],'movement_mean':g['normalized_symmetric_difference']['mean'],
                'composition_ci95':None,'movement_ci95':None}
        # Undefined draws are counted, not dropped/imputed or replaced by an
        # alternative bootstrap. Any such draw makes this interval unavailable.
        if not undefined:
            a=(moved & eligible)[indices].sum(1)/denom
            b=(movement*eligible)[indices].sum(1)/denom
            record['composition_ci95']=np.quantile(a,[.025,.975]).tolist()
            record['movement_ci95']=np.quantile(b,[.025,.975]).tolist()
        results.append(record)
    result={'at':datetime.now(timezone.utc).isoformat(),'operation_id':'shil-novelty-evidence-expansion-20260904',
      'host':summary['host'],'source_csv_sha256':sha(base/'analysis/paired_endpoints.csv'),'script_sha256':sha(Path(__file__)),
      'scope':'pointwise descriptive seed-bootstrap; joint seed index matrix shared by all groups; no superiority/equivalence inference',
      'missingness_rule':'do not discard undefined ratio-bootstrap replicates; if any denominator is zero, withhold interval and retain counts/point summary','groups':results}
    if out.exists():raise ValueError('OUTPUT_ALREADY_EXISTS')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({'groups':len(results),'intervals_available':sum(x['ci_status']=='available' for x in results)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();summarize(a.base,a.output)
