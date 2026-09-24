"""Summarize every frozen CPSS grid point without selecting a new test policy."""
from pathlib import Path
import argparse,csv,json,hashlib
from datetime import datetime,timezone
from analysis_evidence_gate import validate
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def yes(x):return x=='True'
def csvread(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 validate(a.base,phase='science',check_grid=True)
 s=read(a.base/'analysis/summary.json');cfg=read(ROOT/'config/extension.json');proof=read(a.base/'provenance_verification.json')
 if s['phase']!='science' or proof['status']!='PASS' or s['status']!='complete_verified_descriptive':raise ValueError('COMPLETE_SCIENCE_ONLY')
 seeds=cfg['seeds'] if s['host']=='mta' else cfg['replication_seeds'];grid=csvread(a.base/'analysis/grid_diagnostics.csv');pairs=csvread(a.base/'analysis/paired_endpoints.csv')
 if len(grid)!=8*len(seeds)*4*24 or len(pairs)!=8*len(seeds)*4:raise ValueError('INCOMPLETE_GRID')
 qgrid=cfg['resolved_config']['policies.cpss_one_se']['q_grid'];pigrid=cfg['resolved_config']['policies.cpss_one_se']['pi_grid']
 pair_lookup={(x['unit_id'],x['ranker_id']):x for x in pairs};point_rows=[];cell_rows=[]
 for ranker in ['shil','l1','tree','screen']:
  for spec in cfg['conditions']:
   condition=spec['id'];rows=[x for x in grid if x['ranker_id']==ranker and x['condition']==condition]
   for phase in ['cpss_tuning','cpss_final']:
    for q in qgrid:
     for pi in pigrid:
      rr=[x for x in rows if x['phase']==phase and int(x['q'])==q and float(x['pi'])==pi]
      if len(rr)!=len(seeds) or {int(x['seed']) for x in rr}!=set(seeds):raise ValueError('GRID_POINT_SEED_SET')
      eligible=[x for x in rr if yes(x['eligible']) and yes(x['nonempty'])]
      point_rows.append({'ranker_id':ranker,'condition':condition,'phase':phase,'q':q,'pi':pi,'cells':len(rr),
        'empty':sum(not yes(x['nonempty']) for x in rr),'eligible_nonempty':len(eligible),'changed_eligible_nonempty':sum(yes(x['changed']) for x in eligible),
        'selected_by_validation':sum(yes(x['chosen_parameter']) for x in rr)})
   for seed in seeds:
    uid=f'{condition}_{seed}';rr=[x for x in rows if int(x['seed'])==seed and x['phase']=='cpss_final'];pair=pair_lookup[(uid,ranker)]
    eligible=[x for x in rr if yes(x['eligible']) and yes(x['nonempty'])];changed=[x for x in eligible if yes(x['changed'])]
    cell_rows.append({'unit_id':uid,'seed':seed,'condition':condition,'ranker_id':ranker,'final_grid_points':len(rr),'eligible_nonempty_grid_points':len(eligible),
      'changed_eligible_nonempty_grid_points':len(changed),'any_changed_final_grid_point':bool(changed),
      'selected_nonempty_comparable':yes(pair['eligible']) and yes(pair['nonempty']),
      'selected_support_changed':yes(pair['eligible']) and yes(pair['nonempty']) and yes(pair['changed']),
      'unchosen_changed_point_exists':any(not yes(x['chosen_parameter']) for x in changed)})
 if a.output.exists():raise ValueError('OUTPUT_ALREADY_EXISTS')
 a.output.mkdir(parents=True);csvwrite(a.output/'all_grid_point_counts.csv',point_rows);csvwrite(a.output/'per_cell_grid_identity.csv',cell_rows)
 groups=[]
 for ranker in ['shil','l1','tree','screen']:
  for spec in cfg['conditions']:
   rr=[x for x in cell_rows if x['ranker_id']==ranker and x['condition']==spec['id']]
   groups.append({'ranker_id':ranker,'condition':spec['id'],'cells':len(rr),'any_changed_final_grid_point':sum(x['any_changed_final_grid_point'] for x in rr),
     'selected_support_changed':sum(x['selected_support_changed'] for x in rr),'unchosen_changed_but_selected_unchanged_or_empty':sum(x['unchosen_changed_point_exists'] and not x['selected_support_changed'] for x in rr)})
 report={'at':datetime.now(timezone.utc).isoformat(),'operation_id':'shil-novelty-evidence-expansion-20260904','host':s['host'],'groups':groups,
   'grid_points':'all12 q/pi combinations, both phases, every frozen seed/condition/ranker',
   'interpretation':'counts describe whether selected-point identity extends to the evaluated grid; no test-optimal q/pi chosen and no unselected-grid predictive benefit claimed',
   'script_sha256':sha(Path(__file__)),'grid_csv_sha256':sha(a.base/'analysis/grid_diagnostics.csv'),'paired_csv_sha256':sha(a.base/'analysis/paired_endpoints.csv')}
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'host':s['host'],'grid_point_rows':len(point_rows),'cell_rows':len(cell_rows)}))
if __name__=='__main__':main()
