"""Post hoc term-level description of verified selected-support mismatches."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,csv,collections,hashlib
import analysis_evidence_gate as gate

def describe(base,output):
 summary,proof,_=gate.validate(base,phase='science')
 if output.exists():raise ValueError('OUTPUT_ALREADY_EXISTS')
 rows=[]
 for unit in proof['units']:
  d=gate.read(base/'units'/unit['unit_id']/'result.json.gz')
  truth={tuple(t) for t in d['true_edges']}
  groups=[{tuple(t) for t in g} for g in d['proxy_groups']] or [{t} for t in sorted(truth)]
  edges=[tuple(t) for t in d['diagnostic']['candidate_edges']]
  term_group={t:i for i,g in enumerate(groups) for t in g}
  if sum(len(g) for g in groups)!=len(term_group):raise ValueError('OVERLAPPING_ORACLE_GROUPS')
  for pair in d['diagnostic']['pairs']:
   S={edges[i] for i in pair['cpss_support']};T={edges[i] for i in pair['raw_support']}
   if not pair['eligible'] or not pair['nonempty'] or S==T:continue
   gs={term_group[t] for t in S if t in term_group};gt={term_group[t] for t in T if t in term_group}
   k=len(S);ng=len(groups)
   delta=2*(len(gs)-len(gt))/(k+ng)
   if abs(delta-pair['delta_group_f1'])>1e-12:raise ValueError('GROUP_COVERAGE_ACCOUNTING')
   delta_exact=2*(len(S&truth)-len(T&truth))/(k+len(truth))
   if abs(delta_exact-pair['delta_exact_f1'])>1e-12:raise ValueError('EXACT_COVERAGE_ACCOUNTING')
   category='same_covered_groups' if gs==gt else 'different_groups_same_count' if len(gs)==len(gt) else 'covered_group_count_changed'
   def details(terms):
    return [{'term':list(t),'exact_truth':t in truth,'oracle_group':term_group.get(t)} for t in sorted(terms)]
   rows.append({'unit_id':d['unit_id'],'condition':d['spec']['id'],'seed':d['seed'],'ranker_id':pair['ranker_id'],'k':k,'category':category,'removed':details(T-S),'added':details(S-T),'raw_covered_groups':sorted(gt),'cpss_covered_groups':sorted(gs),'outside_declared_groups_changed':sum(t not in term_group for t in S^T),'delta_exact_f1':pair['delta_exact_f1'],'delta_group_f1':pair['delta_group_f1'],'delta_log_loss':pair['delta_log_loss'],'source_result_sha256':unit['result_sha256']})
 output.mkdir(parents=True)
 (output/'changed_terms.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
 count=collections.Counter(r['category'] for r in rows)
 report={'status':'VERIFIED_DESCRIPTIVE_READBACK','at':datetime.now(timezone.utc).isoformat(),'operation_id':'shil-novelty-evidence-expansion-20260904','host':summary['host'],'changed_comparisons':len(rows),'category_counts':dict(count),'changed_terms_outside_oracle_groups':sum(x['outside_declared_groups_changed'] for x in rows),'provenance_sha256':gate.sha(base/'provenance_verification.json'),'data_sha256':gate.sha(output/'changed_terms.json'),'script_sha256':gate.sha(Path(__file__)),'timing':'Taxonomy developed after VPS005 outcome inspection, before MTA006 aggregate inspection; post hoc explanatory description, not an additional primary endpoint or hypothesis test.','limits':['Equal group counts do not imply the same covered groups or selected terms.','A proxy-group member can be an exact false inclusion, and a redundant same-group member can be a group false inclusion under one-to-one scoring.','same_covered_groups describes the entire selected support; it need not mean a single pure proxy-for-proxy swap.','No causal or predictive equivalence follows from oracle grouping.']}
 (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();describe(a.base,a.output)
