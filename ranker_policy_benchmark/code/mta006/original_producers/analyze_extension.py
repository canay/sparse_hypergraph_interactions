"""Recompute paired endpoints from saved supports/probabilities; no model fits."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, math
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(p):
    b=Path(p).read_bytes();return json.loads(gzip.decompress(b) if str(p).endswith('.gz') else b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f1(selected,truth):
    s=set(map(tuple,selected));t=set(map(tuple,truth));hit=len(s&t)
    return 2*hit/(len(s)+len(t)) if len(s)+len(t) else 1.0
def gf1(selected,truth,groups):
    s=sorted(set(map(tuple,selected)))
    gg=[set(map(tuple,g)) for g in groups] if groups else [{tuple(t)} for t in truth]
    # Explicit maximum one-to-one matching, independent of the vendor metric.
    assigned={}
    def visit(i,seen):
        for j,g in enumerate(gg):
            if s[i] in g and j not in seen:
                seen.add(j)
                if j not in assigned or visit(assigned[j],seen):assigned[j]=i;return True
        return False
    hits=sum(visit(i,set()) for i in range(len(s)))
    return 2*hits/(len(s)+len(gg)) if len(s)+len(gg) else 1.0
def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def check_close(a,b,label):
    if a is None or b is None:
        if a!=b:raise ValueError(label)
    elif not math.isclose(a,b,rel_tol=0,abs_tol=1e-12):raise ValueError(f'{label}: {a} != {b}')

def analyze(base,host,phase,tag):
    cfg=read(ROOT/'config/extension.json');seeds=cfg['seeds'] if host=='mta' else cfg['replication_seeds']
    expected={f"{s['id']}_{k}" for k in seeds for s in cfg['conditions']}
    if phase=='smoke':expected={f'E0{i+1}_{s}' for i,s in enumerate(cfg['smoke_seeds'])}
    files={p.parent.name:p for p in (base/'units').glob('*/complete.json')}
    if set(files)!=expected:raise ValueError(f'INCOMPLETE_OR_EXTRA_GRID: {len(files)}/{len(expected)}')
    summaries=[];gridrows=[];inputs=[];envs=[];scientific_hashes=[];warning_rows=[]
    for uid in sorted(expected):
        receipt=read(files[uid]);folder=files[uid].parent
        for a in receipt['artifacts']:
            if sha(folder/a['path'])!=a['sha256']:raise ValueError('HASH_DRIFT '+uid+'/'+a['path'])
        inputs.append({'unit_id':uid,'receipt_sha256':sha(files[uid]),'result_sha256':sha(folder/'result.json.gz')})
        identity=read(folder/'identity.json');envs.append(identity['environment'])
        d=read(folder/'result.json.gz');edges=d['diagnostic']['candidate_edges'];truth=d['true_edges'];groups=d['proxy_groups']
        # Read original sub-fit artifacts rather than trusting precomputed grid summaries.
        ranks={};rawranks={};expectedfits=4*(4*(1 if phase=='smoke' else cfg['n_pairs'])+2)
        fitfiles=[a for a in receipt['artifacts'] if a['path'].startswith('fits/')]
        if len(fitfiles)!=expectedfits:raise ValueError('FIT_COUNT '+uid)
        for a in fitfiles:
            f=read(folder/a['path']);b=f['binding'];r=f['result']
            if b['phase']=='final_development_rank':rawranks[b['ranker']]=r['ranked_indices']
            ranks.setdefault((b['ranker'],b['phase']),[]).append(r['ranked_indices'])
            for warning in r.get('warnings',[]):warning_rows.append({'unit_id':uid,'ranker':b['ranker'],'phase':b['phase'],'warning':warning})
        for row in d['diagnostic']['grid']:
            ranker=row['ranker_id'];rr=ranks[(ranker,row['phase'])]
            freq=np.zeros(len(edges))
            for ranking in rr:freq[np.array(ranking[:row['q']],dtype=int)]+=1
            freq/=len(rr);S=np.flatnonzero(freq>=row['pi']).tolist();T=rawranks[ranker][:len(S)]
            if set(S)!=set(row['cpss_support']) or set(T)!=set(row['raw_support']):raise ValueError('GRID_RECOMPUTE '+uid)
            exact_grid=f1([edges[i] for i in S],truth)-f1([edges[i] for i in T],truth) if len(S)==len(T) else None
            group_grid=gf1([edges[i] for i in S],truth,groups)-gf1([edges[i] for i in T],truth,groups) if len(S)==len(T) else None
            check_close(exact_grid,row['delta_exact_f1'],'grid exact');check_close(group_grid,row['delta_group_f1'],'grid group')
            gridrows.append({'unit_id':uid,'condition':d['spec']['id'],'seed':d['seed'],'ranker_id':ranker,
              'phase':row['phase'],'q':row['q'],'pi':row['pi'],'chosen_parameter':row['chosen_parameter'],
              'k':len(S),'eligible':len(S)==len(T),'nonempty':bool(S),'changed':set(S)!=set(T),
              'delta_exact_f1':exact_grid,'delta_group_f1':group_grid})
        for pair in d['diagnostic']['pairs']:
            S=pair['cpss_support'];T=pair['raw_support'];eligible=len(S)==len(T)
            if set(T)!=set(rawranks[pair['ranker_id']][:len(S)]):raise ValueError('RAW_REFERENCE '+uid)
            es=[edges[i] for i in S];et=[edges[i] for i in T]
            exact=f1(es,truth)-f1(et,truth) if eligible else None
            group=gf1(es,truth,groups)-gf1(et,truth,groups) if eligible else None
            y=np.array(d['diagnostic']['test_labels'],int);loss={}
            for label in ('cpss','raw'):
                probs=np.array(pair['probabilities'][label],float)
                if probs.shape!=(len(y),2) or not np.isfinite(probs).all() or np.any(probs<0) or not np.allclose(probs.sum(1),1):raise ValueError('INVALID_PROBABILITIES')
                losses=-np.log(np.clip(probs[np.arange(len(y)),y],1e-15,1))
                if not np.allclose(losses,pair['per_observation_log_loss'][label],atol=1e-12,rtol=0):raise ValueError('LOSS_VECTOR')
                loss[label]=float(losses.mean())
            delta=loss['cpss']-loss['raw'] if eligible else None
            check_close(exact,pair['delta_exact_f1'],'exact');check_close(group,pair['delta_group_f1'],'group');check_close(delta,pair['delta_log_loss'],'prediction')
            diff=len(set(S)^set(T))
            summaries.append({'unit_id':uid,'condition':d['spec']['id'],'seed':d['seed'],'ranker_id':pair['ranker_id'],
              'n':d['spec']['n'],'proxy':d['spec']['kind']=='redundant_mixed','signal_scale':d['spec']['signal_scale'],
              'k':len(S),'raw_k':len(T),'eligible':eligible,'nonempty':bool(S),'changed':diff>0,
              'symmetric_difference':diff,'movement_fraction':diff/(2*len(S)) if S and eligible else None,
              'chosen_q':pair['chosen_q'],'chosen_pi':pair['chosen_pi'],'delta_exact_f1':exact,
              'delta_group_f1':group,'cpss_log_loss':loss['cpss'],'raw_log_loss':loss['raw'],'delta_log_loss':delta})
        scientific={k:d[k] for k in ('unit_id','spec','seed','dataset_sha256','true_edges','proxy_groups','diagnostic','fixed_k_identity_checks')}
        scientific_hashes.append({'unit_id':uid,'digest':hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()})
    out=base/'analysis';out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Analysis already exists; preserve prior before explicit recomputation')
    csvwrite(out/'paired_endpoints.csv',summaries);csvwrite(out/'grid_diagnostics.csv',gridrows);csvwrite(out/'warnings.csv',warning_rows)
    rng=np.random.default_rng(cfg['bootstrap_seed']);sample_indices=rng.integers(0,len(seeds),size=(cfg['bootstrap_replicates'],len(seeds)))
    groups_out=[]
    for ranker in sorted({r['ranker_id'] for r in summaries}):
        for condition in sorted({r['condition'] for r in summaries}):
            rows=[r for r in summaries if r['ranker_id']==ranker and r['condition']==condition]
            rec={'ranker_id':ranker,'condition':condition,'cells':len(rows),'eligible':sum(r['eligible'] for r in rows),
                 'nonempty':sum(r['nonempty'] for r in rows),'eligible_nonempty':sum(r['nonempty'] and r['eligible'] for r in rows),
                 'changed_nonempty':sum(r['changed'] and r['nonempty'] and r['eligible'] for r in rows),
                 'empty':sum(not r['nonempty'] for r in rows),'endpoints':{}}
            rec['composition_change_fraction']=rec['changed_nonempty']/rec['eligible_nonempty'] if rec['eligible_nonempty'] else None
            movement=np.array([r['movement_fraction'] for r in rows if r['movement_fraction'] is not None])
            rec['normalized_symmetric_difference']={'n':len(movement),'mean':float(movement.mean()) if len(movement) else None,
                                                   'median':float(np.median(movement)) if len(movement) else None}
            for endpoint in ('delta_exact_f1','delta_group_f1','delta_log_loss'):
                values=np.array([r[endpoint] for r in rows if r[endpoint] is not None],float)
                e={'n':len(values),'mean':float(values.mean()) if len(values) else None,
                   'median':float(np.median(values)) if len(values) else None,'ci95':None,'constant_observed':bool(len(values) and np.all(values==values[0]))}
                if phase=='science' and len(values)==len(seeds):
                    lookup={r['seed']:r[endpoint] for r in rows};ordered=np.array([lookup[s] for s in seeds]);means=ordered[sample_indices].mean(1)
                    e['ci95']=np.quantile(means,[.025,.975]).tolist()
                rec['endpoints'][endpoint]=e
            groups_out.append(rec)
    summary={'schema_version':1,'at':datetime.now(timezone.utc).isoformat(),'host':host,'phase':phase,
             'status':'complete_verified_descriptive','planned_cells':len(expected),'observed_cells':len(files),
             'paired_rows':len(summaries),'grid_rows':len(gridrows),'warning_count':len(warning_rows),
             'bound_freeze_sha256':sha(ROOT/'FREEZE.json'),'groups':groups_out,
             'uncertainty':'pointwise descriptive seed-bootstrap; no superiority/equivalence inference',
             'environments':list({json.dumps(e,sort_keys=True):e for e in envs}.values())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'scientific_payload_digests.json').write_text(json.dumps(scientific_hashes,indent=2),encoding='utf-8')
    verdict={'status':'PASS','at':summary['at'],'verified_artifacts':[{'path':str((out/'summary.json').relative_to(ROOT)),'sha256':sha(out/'summary.json')},
             {'path':str((out/'paired_endpoints.csv').relative_to(ROOT)),'sha256':sha(out/'paired_endpoints.csv')}],
             'raw_input_digest':{'cell_count':len(inputs),'per_file_hashes':inputs},
             'analyzer_sha256':sha(__file__),'checks':['receipt hashes','fit counts','raw rank identity','all-grid frequency reconstruction',
                 'independent exact/group F1','independent test log loss'],'scientific_verdict':'descriptive_only'}
    (out/'independent_recompute.json').write_text(json.dumps(verdict,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('status','observed_cells','paired_rows','grid_rows','warning_count')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',choices=['mta','vps'],required=True);p.add_argument('--phase',choices=['smoke','science'],default='science');p.add_argument('--tag',default='baseline')
    a=p.parse_args();analyze(ROOT/'outputs'/a.host/a.phase/a.tag,a.host,a.phase,a.tag)
