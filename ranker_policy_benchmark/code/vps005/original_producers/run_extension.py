"""MCH-SHIL-005: frozen, resumable diagnostic using byte-preserved v6 rankers."""
from __future__ import annotations
import argparse, copy, ctypes, dataclasses, gzip, hashlib, importlib.metadata
import json, math, os, platform, signal, socket, subprocess, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
import psutil
import sc_shil_experiment as legacy
import track_a_runner as runner
from ranker_adapters import registry_ranker_adapter
from ranker_protocol import RankingResult, ScoreProvenance, validate_ranking_result
from method_registry import load_method_registry
from support_metrics import support_metrics

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(x):
    if dataclasses.is_dataclass(x): return clean(dataclasses.asdict(x))
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,set): return [clean(v) for v in sorted(x)]
    if isinstance(x,float) and not math.isfinite(x): return None
    return x
def encoded(x): return json.dumps(clean(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x): return hashlib.sha256(encoded(x)).hexdigest()
def atomic(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=encoded(payload)
    if path.suffix=='.gz': data=gzip.compress(data,compresslevel=3,mtime=0)
    tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('xb') as f: f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    if os.name=='posix':
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
def read(path):
    data=Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if Path(path).suffix=='.gz' else data)
def env():
    out={'os':platform.system(),'architecture':platform.machine(),'python':platform.python_version(),
         'hostname':socket.gethostname(),'executable':sys.executable,'packages':{}}
    for pkg in ('numpy','scipy','scikit-learn','pandas','psutil'):
        out['packages'][pkg]=importlib.metadata.version(pkg)
    return out
def config():return read(ROOT/'config/extension.json')
def verify_freeze():
    f=read(ROOT/'FREEZE.json')
    for item in f['files']:
        if sha(ROOT/item['path'])!=item['sha256']:raise RuntimeError('FROZEN_INPUT_DRIFT '+item['path'])
    return sha(ROOT/'FREEZE.json')
def lock():
    paths=[ROOT/'run_extension.py',ROOT/'analyze_extension.py',ROOT/'PROTOCOL.md',
           ROOT/'config/extension.json',ROOT/'config/method_registry.json',ROOT/'vendor_manifest.json',ROOT/'technical_preflight.py']
    paths+=sorted((ROOT/'vendor').glob('*.py'))
    out={'schema_version':1,'created_at':now(),'change_id':'MCH-SHIL-005',
         'tool':'Codex','model':'gpt-5.6-sol xhigh','operation_id':config()['operation_id'],
         'status':'prospective_inputs_locked','files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]}
    if (ROOT/'FREEZE.json').exists():raise RuntimeError('Freeze already exists; preserve it before authorized code correction')
    atomic(ROOT/'FREEZE.json',out);print('LOCKED',sha(ROOT/'FREEZE.json'),flush=True)

def make_data(spec,seed):
    """Keep y identical when only proxy availability changes."""
    n,d=int(spec['n']),int(spec['d'])
    base_rng,proxy_rng,noise_rng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    X=base_rng.normal(size=(n,d))
    residual=proxy_rng.normal(size=(n,4))
    noise=noise_rng.normal(scale=spec['noise_sd'],size=n)
    edges,coefs=legacy._base_edges('mixed')
    score=sum(spec['signal_scale']*c*np.prod(X[:,e],axis=1) for e,c in zip(edges,coefs))
    score=score+0.15*X[:,14]-0.12*X[:,15]+noise
    y=(score>np.quantile(score,spec['threshold_quantile'])).astype(int)
    groups=[]
    if spec['kind']=='redundant_mixed':
        rho=spec['proxy_correlation']
        for j,(proxy,source) in enumerate([(16,0),(17,1),(18,4),(19,5)]):
            X[:,proxy]=rho*X[:,source]+math.sqrt(1-rho*rho)*residual[:,j]
        groups=[{(0,1),(1,16),(0,17),(16,17)},{(2,3)},
                {(4,5,6),(5,6,18),(4,6,19),(6,18,19)},{(7,8,9)}]
    return legacy.Dataset(spec['id'],X,y,[f'x{i}' for i in range(d)],
                          {tuple(e) for e in edges},groups,{**spec,'seed':seed,'generator':'controlled_proxy_v1'})

def cases(host,phase):
    cfg=config()
    if phase=='smoke': return [(cfg['conditions'][i],s) for i,s in enumerate(cfg['smoke_seeds'])]
    seeds=cfg['seeds'] if host=='mta' else cfg['replication_seeds']
    # Interleave conditions inside each seed; never sort by an outcome.
    return [(spec,s) for s in seeds for spec in cfg['conditions']]
def effective(spec,phase):
    cfg=config();spec=copy.deepcopy(spec);resolved=copy.deepcopy(cfg['resolved_config'])
    B=cfg['n_pairs']
    if phase=='smoke':
        spec['n']=80;B=1
        resolved['rankers.shil'].update(epochs=3,patience=2)
        resolved['rankers.tree']['n_estimators']=10
        resolved['rankers.l1']['c_grid']=[0.05,0.25]
    return spec,resolved,B
def unit_id(spec,seed):return f"{spec['id']}_{seed}"
def phase_root(args):return ROOT/'outputs'/args.host/args.phase/args.tag

class CachedAdapter:
    def __init__(self,registry,ranker_id,folder,fingerprint,observed,abort_after):
        self.base=registry_ranker_adapter(registry,ranker_id)
        self.ranker=next(r for r in registry.rankers if r.ranker_id==ranker_id)
        self.ranker_id=ranker_id;self.folder=folder;self.fingerprint=fingerprint
        self.observed=observed;self.abort_after=abort_after;self.index=0
    def fit(self,request):
        index=self.index;self.index+=1
        key=f'{self.ranker_id}_{index:03d}'
        path=self.folder/'fits'/(key+'.json.gz')
        binding={'unit':self.fingerprint,'ranker':self.ranker_id,'index':index,'phase':request.phase,
                 'seed':request.model_seed,'split':request.split_plan_sha256,
                 'config':request.config_sha256,'universe':request.candidate_universe_sha256,
                 'fit':digest(request.fit_indices),'validation':digest(request.validation_indices)}
        cached=False
        if path.exists():
            obj=read(path)
            if obj['binding']!=clean(binding) or digest(obj['result'])!=obj['result_sha256']:
                raise RuntimeError('FIT_CACHE_DRIFT '+key)
            r=obj['result'];r['provenance']=ScoreProvenance(**r['provenance'])
            for field in ('scores','ranked_indices','active_indices'):
                r[field]=np.array(r[field],dtype=float if field=='scores' else np.int64)
            r['warnings']=tuple(r['warnings']);result=RankingResult(**r);cached=True
        else:
            atomic(self.folder/'progress.json',{'at':now(),'pid':os.getpid(),'phase':request.phase,
                   'ranker':self.ranker_id,'fit_index':index,'completed_fit_count':len(self.observed)})
            result=self.base.fit(request)
            payload=clean(result)
            atomic(path,{'binding':binding,'result':payload,'result_sha256':digest(payload)})
        result=validate_ranking_result(result,request,self.ranker)
        self.observed.append({'ranker_id':self.ranker_id,'index':index,'phase':request.phase,
              'ranked_indices':result.ranked_indices.tolist(),'path':path.relative_to(self.folder).as_posix(),
              'sha256':sha(path),'cached':cached})
        atomic(self.folder/'progress.json',{'at':now(),'pid':os.getpid(),'phase':request.phase,
               'ranker':self.ranker_id,'fit_index':index,'completed_fit_count':len(self.observed),'cache_hit':cached})
        if self.abort_after and len(self.observed)==self.abort_after:
            print('CONTROLLED_INTERRUPTION_AFTER_FITS',len(self.observed),flush=True)
            os._exit(86)
        return result

def comparison(S,T,edges,ds):
    S=tuple(sorted(S));T=tuple(sorted(T));k=len(S);diff=len(set(S)^set(T))
    sm=support_metrics([edges[i] for i in S],ds.true_edges,ds.equivalence_groups)
    tm=support_metrics([edges[i] for i in T],ds.true_edges,ds.equivalence_groups)
    return {'cpss_support':S,'raw_support':T,'k':k,'raw_k':len(T),'eligible':len(T)==k,
       'nonempty':bool(k),'identical':S==T,'symmetric_difference':diff,
       'movement_fraction':diff/(2*k) if k and len(T)==k else None,
       'cpss_metrics':sm,'raw_metrics':tm,
       'delta_exact_f1':sm['support_f1']-tm['support_f1'] if len(T)==k else None,
       'delta_group_f1':sm['group_f1']-tm['group_f1'] if len(T)==k else None}

def evaluate_extra(ds,result,observed,seed,cfg):
    raw={r['ranker_id']:r for r in result.ranking_scores if r['score_source']=='final_ranker_score'}
    selected={r['ranker_id']:r for r in result.selected_edges if r['policy_id']=='cpss_one_se'}
    chosen={r['ranker_id']:r for r in result.metrics if r['policy_id']=='cpss_one_se'}
    edges=runner.candidate_edges_for_orders(ds.X.shape[1],cfg['candidate_orders'])
    X,split=runner._outer_scaled_matrix(ds,seed)
    dev=np.sort(np.r_[split['train_idx'],split['val_idx']]);test=np.asarray(split['test_idx'])
    pairs=[];grid=[];frequencies=[]
    for ranker,r in raw.items():
        ranking=r['ranked_indices'];S=selected[ranker]['support_indices'];T=ranking[:len(S)]
        pair={'ranker_id':ranker,**comparison(S,T,edges,ds),
              'chosen_q':chosen[ranker]['chosen_q'],'chosen_pi':chosen[ranker]['chosen_pi']}
        probs={};losses={}
        for label,support in [('cpss',S),('raw',T)]:
            sup=[edges[i] for i in sorted(support)]
            model=legacy.fit_l2_refit(X[dev],ds.y[dev],sup,cfg['interaction_clip'],seed)
            p=model.predict_proba(legacy.refit_design(X[test],sup,cfg['interaction_clip']))
            probs[label]=p;losses[label]=legacy.per_observation_log_loss(ds.y[test],p,model.classes_)
        pair.update(probabilities=probs,per_observation_log_loss=losses,
                    delta_log_loss=float(np.mean(losses['cpss']-losses['raw'])) if pair['eligible'] else None)
        if pair['identical'] and not np.allclose(probs['cpss'],probs['raw'],rtol=0,atol=1e-12):
            raise RuntimeError('IDENTICAL_SUPPORT_REFIT_MISMATCH')
        pairs.append(pair)
        for phase in ('cpss_tuning','cpss_final'):
            ranks=[o['ranked_indices'] for o in observed if o['ranker_id']==ranker and o['phase']==phase]
            expected=2*(1 if cfg.get('technical_smoke') else cfg['n_pairs'])
            if len(ranks)!=expected:raise RuntimeError('HALF_FIT_COUNT_MISMATCH')
            for q in cfg['resolved_config']['policies.cpss_one_se']['q_grid']:
                f=np.zeros(len(edges))
                for ranking_half in ranks:f[np.asarray(ranking_half[:q],dtype=int)]+=1
                f/=len(ranks)
                frequencies.append({'ranker_id':ranker,'phase':phase,'q':q,'frequencies':f})
                for pi in cfg['resolved_config']['policies.cpss_one_se']['pi_grid']:
                    supp=np.flatnonzero(f>=pi).tolist();rr=ranking[:len(supp)]
                    grid.append({'ranker_id':ranker,'phase':phase,'q':q,'pi':pi,
                       'chosen_parameter':q==pair['chosen_q'] and pi==pair['chosen_pi'],
                       **comparison(supp,rr,edges,ds)})
                    if phase=='cpss_final' and q==pair['chosen_q'] and pi==pair['chosen_pi']:
                        if set(supp)!=set(S):raise RuntimeError('RECONSTRUCTED_CPSS_MISMATCH')
    return {'pairs':pairs,'grid':grid,'frequencies':frequencies,'test_indices':test,
            'test_labels':ds.y[test],'development_indices':dev,'candidate_edges':edges}

def verify_unit(folder,fingerprint=None):
    receipt=read(folder/'complete.json')
    if fingerprint and receipt['fingerprint']!=fingerprint:raise RuntimeError('UNIT_INPUT_DRIFT')
    for item in receipt['artifacts']:
        if sha(folder/item['path'])!=item['sha256']:raise RuntimeError('UNIT_ARTIFACT_DRIFT '+item['path'])
    return receipt

def worker(args):
    # Terminate this worker if its Linux supervisor dies; no detached descendants.
    if sys.platform.startswith('linux'):
        parent=os.getppid();ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=parent:raise RuntimeError('Supervisor vanished during startup')
    freeze_sha=verify_freeze();cfg=config()
    spec=next(s for s in cfg['conditions'] if s['id']==args.condition)
    spec,resolved,B=effective(spec,args.phase)
    ds=make_data(spec,args.seed)
    dataset_hash=hashlib.sha256(ds.X.tobytes()+ds.y.tobytes()).hexdigest()
    fingerprint=digest({'freeze':freeze_sha,'spec':spec,'seed':args.seed,'phase':args.phase,
                        'resolved':resolved,'B':B,'data':dataset_hash,'environment':env()})
    folder=phase_root(args)/'units'/unit_id(spec,args.seed);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        verify_unit(folder,fingerprint);print('VERIFIED_REUSE',folder.name,flush=True);return
    if (folder/'identity.json').exists() and read(folder/'identity.json')['fingerprint']!=fingerprint:
        raise RuntimeError('PARTIAL_UNIT_INPUT_DRIFT')
    atomic(folder/'identity.json',{'fingerprint':fingerprint,'dataset_sha256':dataset_hash,'freeze_sha256':freeze_sha,
         'condition':spec,'seed':args.seed,'environment':env(),'phase':args.phase})
    observed=[];started=time.monotonic();registry=load_method_registry(ROOT/'config/method_registry.json')
    def factory(registry,ranker_id):
        return CachedAdapter(registry,ranker_id,folder,fingerprint,observed,args.abort_after)
    result=runner.run_track_a_cell(ds,registry=registry,resolved_config=resolved,
        cell_metadata={'cell_type':'synthetic','candidate_orders':cfg['candidate_orders']},
        interaction_clip=cfg['interaction_clip'],outer_split_seed=args.seed,master_seed=args.seed,
        n_pairs=B,adapter_factory=factory)
    expected=4*(4*B+2)
    if len(observed)!=expected:raise RuntimeError('FIT_BUDGET_MISMATCH')
    cfg['technical_smoke']=args.phase=='smoke'
    extra=evaluate_extra(ds,result,observed,args.seed,cfg)
    fixed=0
    raw={r['ranker_id']:r for r in result.ranking_scores if r['score_source']=='final_ranker_score'}
    metrics={r['method_id']:r for r in result.metrics}
    for row in result.selected_edges:
        if row['policy_id']=='fixed_k':
            k=int(metrics[row['method_id']]['requested_k'])
            if set(row['support_indices'])!=set(raw[row['ranker_id']]['ranked_indices'][:k]):
                raise RuntimeError('FIXED_K_IDENTITY_FAILURE')
            fixed+=1
    payload={'schema_version':1,'unit_id':folder.name,'fingerprint':fingerprint,'spec':spec,'seed':args.seed,
             'dataset_sha256':dataset_hash,'true_edges':ds.true_edges,'proxy_groups':ds.equivalence_groups,
             'track_a':result,'diagnostic':extra,'fixed_k_identity_checks':fixed}
    atomic(folder/'result.json.gz',payload)
    artifacts=[{'path':'result.json.gz','sha256':sha(folder/'result.json.gz')}]
    artifacts += [{'path':o['path'],'sha256':o['sha256']} for o in observed]
    atomic(folder/'complete.json',{'schema_version':1,'status':'complete','unit_id':folder.name,
          'fingerprint':fingerprint,'at':now(),'elapsed_seconds':time.monotonic()-started,
          'fit_count':expected,'cache_hits':sum(o['cached'] for o in observed),
          'artifact_bytes':sum((folder/a['path']).stat().st_size for a in artifacts),'artifacts':artifacts})
    verify_unit(folder,fingerprint);print('COMPLETE',folder.name,'seconds',round(time.monotonic()-started,2),flush=True)

def sample_process(proc):
    out={'pid':proc.pid,'alive':proc.poll() is None,'cpu_seconds':0.0,'rss_bytes':0,'io_bytes':0}
    try:
        p=psutil.Process(proc.pid)
        for child in [p]+p.children(recursive=True):
            c=child.cpu_times();out['cpu_seconds']+=c.user+c.system;out['rss_bytes']+=child.memory_info().rss
            try:i=child.io_counters();out['io_bytes']+=i.read_bytes+i.write_bytes
            except (psutil.Error,AttributeError):pass
    except psutil.Error:pass
    return out
def kill_owned(proc):
    if proc.poll() is not None:return
    if os.name=='posix':os.killpg(proc.pid,signal.SIGTERM)
    else:proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name=='posix':os.killpg(proc.pid,signal.SIGKILL)
        else:proc.kill()
        proc.wait(timeout=5)

def supervise(args):
    freeze_sha=verify_freeze();cfg=config();base=phase_root(args);base.mkdir(parents=True,exist_ok=True)
    if args.phase=='science':
        gate=read(ROOT/'PREFLIGHT_PASS.json')
        if gate['freeze_sha256']!=freeze_sha or gate.get('status')!='PASS':raise RuntimeError('PREFLIGHT_NOT_CURRENT')
    workers=args.workers or cfg['max_workers'][args.host]
    if workers>cfg['max_workers'][args.host] or workers<1:raise ValueError('Worker ceiling')
    plan=cases(args.host,args.phase)
    if args.only:plan=[(s,k) for s,k in plan if unit_id(s,k) in args.only.split(',')]
    lockpath=base/'supervisor.lock'
    if lockpath.exists():
        previous=read(lockpath)
        if psutil.pid_exists(previous['pid']):raise RuntimeError('Existing supervisor PID; inspect before resume')
        os.replace(lockpath,base/('stale-lock-'+uuid.uuid4().hex+'.json'))
    # Exclusive create prevents competing supervisors with the same planned paths.
    with lockpath.open('x') as f:json.dump({'pid':os.getpid(),'at':now()},f)
    started=time.monotonic();pending=list(plan);active={};done=[];failed=[];attempt=uuid.uuid4().hex
    atomic(base/('invocation-'+attempt+'.json'),{'at':now(),'args':vars(args),'environment':env(),
        'freeze_sha256':freeze_sha,'planned_units':[unit_id(s,k) for s,k in plan]})
    stopping=False
    def on_signal(signum,frame):
        nonlocal stopping;stopping=True
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    try:
        while pending or active:
            resources={'available_ram_bytes':psutil.virtual_memory().available,
                       'free_disk_bytes':__import__('shutil').disk_usage(base).free,
                       'output_bytes':sum(p.stat().st_size for p in base.rglob('*') if p.is_file())}
            if args.phase=='science' and (resources['free_disk_bytes']<cfg['minimum_disk_bytes'] or
                                         resources['available_ram_bytes']<cfg['minimum_available_ram_bytes'] or
                                         resources['output_bytes']>cfg['maximum_output_bytes']):
                failed.append({'unit':'resource','reason':'RESOURCE_HARD_STOP','resources':resources});stopping=True
            if time.monotonic()-started>cfg['whole_run_timeout_seconds']:
                failed.append({'unit':'watchdog','reason':'WHOLE_RUN_TIMEOUT'});stopping=True
            while pending and len(active)<workers and not stopping:
                spec,seed=pending.pop(0);uid=unit_id(spec,seed);folder=base/'units'/uid
                command=[sys.executable,str(ROOT/'run_extension.py'),'worker','--host',args.host,'--phase',args.phase,
                         '--tag',args.tag,'--condition',spec['id'],'--seed',str(seed)]
                if args.abort_after:command+=['--abort-after',str(args.abort_after)]
                logfile=base/'logs'/f'{uid}-{attempt}.log';logfile.parent.mkdir(exist_ok=True)
                log=logfile.open('xb');p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
                active[uid]={'proc':p,'log':log,'start':time.monotonic(),'progress_at':time.monotonic(),
                             'signature':None,'folder':folder,'command':command}
                print('START',uid,'pid',p.pid,flush=True)
            samples=[]
            for uid,item in list(active.items()):
                p=item['proc'];sample=sample_process(p)
                sample.update(unit_id=uid,run_id=ROOT.name,attempt_id=attempt,timestamp=now(),
                              unit_elapsed_seconds=time.monotonic()-item['start'],worker_pid=p.pid)
                samples.append(sample)
                progress=item['folder']/'progress.json'
                stamp=progress.stat().st_mtime_ns if progress.exists() else 0
                signature=(sample['cpu_seconds'],sample['io_bytes'],stamp)
                if signature!=item['signature']:item['progress_at']=time.monotonic();item['signature']=signature
                reason=None
                if time.monotonic()-item['start']>cfg['per_unit_timeout_seconds']:reason='UNIT_TIMEOUT'
                if time.monotonic()-item['progress_at']>cfg['stall_seconds']:reason='WORKER_STALL'
                if stopping:reason=reason or 'SUPERVISOR_STOP'
                if reason:kill_owned(p)
                code=p.poll()
                if code is not None:
                    item['log'].close()
                    if code==0 and not reason:
                        receipt=verify_unit(item['folder']);done.append(uid)
                        print('VERIFIED',uid,receipt['elapsed_seconds'],flush=True)
                    else:
                        failed.append({'unit':uid,'exit_code':code,'reason':reason or 'WORKER_ERROR'})
                        print('FAILED',uid,code,reason,flush=True);stopping=True
                    del active[uid]
            heartbeat={'at':now(),'timestamp':now(),'run_id':ROOT.name,'attempt_id':attempt,'supervisor_pid':os.getpid(),
              'host':args.host,'phase':args.phase,'complete':len(done),'planned':len(plan),'remaining':len(pending),
              'workers':samples,'resources':resources,'elapsed_seconds':time.monotonic()-started}
            atomic(base/'heartbeat.json',heartbeat)
            with (base/('heartbeat-history-'+attempt+'.jsonl')).open('ab') as f:f.write(encoded(heartbeat)+b'\n')
            if stopping and not active:break
            if active:time.sleep(0.25 if args.phase=='smoke' else cfg['heartbeat_seconds'])
        status={'status':'complete' if len(done)==len(plan) and not failed else 'incomplete',
                'at':now(),'freeze_sha256':freeze_sha,'completed':done,'failed':failed,
                'pending':[unit_id(s,k) for s,k in pending],'elapsed_seconds':time.monotonic()-started,
                'planned':len(plan),'phase':args.phase,'host':args.host,'environment':env()}
        atomic(base/('terminal-'+attempt+'.json'),status);atomic(base/'status.json',status)
        if status['status']!='complete':raise RuntimeError('RUN_INCOMPLETE; preserved attempts and caches')
    finally:
        for item in active.values():kill_owned(item['proc']);item['log'].close()
        if lockpath.exists():os.replace(lockpath,base/('closed-lock-'+attempt+'.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['lock','worker','supervise','invariants'])
    p.add_argument('--host',choices=['mta','vps'],default='mta');p.add_argument('--phase',choices=['smoke','science'],default='smoke')
    p.add_argument('--tag',default='baseline');p.add_argument('--condition',default='E01');p.add_argument('--seed',type=int,default=90001)
    p.add_argument('--workers',type=int);p.add_argument('--abort-after',type=int,default=0);p.add_argument('--only')
    args=p.parse_args()
    if args.command=='lock':lock()
    elif args.command=='worker':worker(args)
    elif args.command=='supervise':supervise(args)
    else:
        checks=[]
        for a,b in zip(config()['conditions'][::2],config()['conditions'][1::2]):
            x=make_data(a,90001);y=make_data(b,90001)
            assert np.array_equal(x.y,y.y) and np.array_equal(x.X[:,:16],y.X[:,:16])
            assert not np.array_equal(x.X[:,16:],y.X[:,16:])
            checks.append({'absent':a['id'],'proxy':b['id'],'label_identity':True,'nonproxy_identity':True})
        print(json.dumps({'status':'PASS','generator_checks':checks,'environment':env()}))
if __name__=='__main__':main()
