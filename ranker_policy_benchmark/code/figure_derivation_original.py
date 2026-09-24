"""ISS-08: read immutable MCH006 summaries; derive two publication carriers.

No scientific fitting, bootstrap resampling, or frozen output modification.
Figure 3 keeps the unique changed/nonempty slice. Figure 4 retains all means
and intervals and prints means to preserve the old heatmaps' numeric lookup.
"""
from pathlib import Path
import csv, json, hashlib, sys
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/2026-09-04_codex_composition-quality-l1-correction'
BASE=EXP/'collected/mta_science_baseline/outputs/mta/science/baseline'
PUB=BASE/'analysis/publication'
OUT=Path(__file__).resolve().parent/'derived_v2'
OP='shil-weekend-round-f-20260905-stage01'
RANKERS=['shil','l1','tree','screen']
LABELS=['SHIL','L1','RF','ANOVA F']
CONDITIONS=[f'E{i:02d}' for i in range(1,9)]
ENDPOINTS=['delta_exact_f1','delta_group_f1','delta_log_loss']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def read(p): return json.loads(p.read_text(encoding='utf-8'))

def main():
    m=read(PUB/'manifest.json')
    assert sha(BASE/'analysis/summary.json')==m['summary_sha256'].upper()
    assert sha(EXP/'prepare_publication_outputs.py')==m['script_sha256'].upper()
    assert all(sha(PUB/a['file'])==a['sha256'].upper() for a in m['artifacts'])
    summary=read(BASE/'analysis/summary.json')
    groups={(g['ranker_id'],g['condition']):g for g in summary['groups']}
    rows=list(csv.DictReader((PUB/'condition_ranker_summary.csv').open(encoding='utf-8')))
    assert len(groups)==len(rows)==32
    assert sum(g['eligible_nonempty'] for g in groups.values())==533
    assert sum(g['changed_nonempty'] for g in groups.values())==54
    numeric=[]
    for row in rows:
        g=groups[row['ranker_id'],row['condition']]
        for endpoint,scale in zip(ENDPOINTS,[100,100,1000]):
            v=g['endpoints'][endpoint]
            assert v['n']==20
            for key in ['mean','median']:
                assert float(row[endpoint+'_'+key])==v[key]
            assert [float(row[endpoint+'_ci_low']),float(row[endpoint+'_ci_high'])]==v['ci95']
            numeric.append({'ranker':row['ranker_id'],'condition':row['condition'],'endpoint':endpoint,
              'mean':v['mean'],'ci95':v['ci95'],'n':20,
              'old_heatmap_scale':scale,'old_heatmap_display':'0' if v['mean']==0 else f"{v['mean']*scale:.2g}",
              'new_unscaled_display':'0' if v['mean']==0 else f"{v['mean']:.2g}"})
    assert not OUT.exists(), 'Use a new version directory; never overwrite a prior derived attempt.'
    OUT.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    matrix=np.array([[groups[r,c]['composition_change_fraction'] for c in CONDITIONS] for r in RANKERS])
    fig,ax=plt.subplots(figsize=(7.15,2.65),layout='constrained')
    im=ax.imshow(matrix,cmap='cividis',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(8),CONDITIONS); ax.set_yticks(range(4),LABELS)
    ax.set_xlabel('Condition'); ax.set_xticks(np.arange(-.5,8,1),minor=True); ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.8);ax.tick_params(which='both',length=0)
    for i,r in enumerate(RANKERS):
        for j,c in enumerate(CONDITIONS):
            g=groups[r,c]; value=matrix[i,j]; color=im.cmap(im.norm(value))
            lum=.2126*color[0]+.7152*color[1]+.0722*color[2]
            ax.text(j,i,f"{g['changed_nonempty']}/{g['eligible_nonempty']}",ha='center',va='center',color='white' if lum<.48 else 'black',fontsize=9)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.08,fraction=.10,aspect=40)
    cb.set_label('Changed fraction among nonempty comparable supports')
    for spine in ax.spines.values(): spine.set_visible(False)
    for ext in ['pdf','png']: fig.savefig(OUT/f'fig_changed_support.{ext}',dpi=600,metadata={'Creator':OP} if ext=='pdf' else None)
    plt.close(fig)

    # Same 96 saved means and 96 saved pointwise intervals. A small numeric
    # column uses the old heatmaps' two significant figures on unscaled axes.
    fig=plt.figure(figsize=(7.15,8.7))
    grid=fig.add_gridspec(1,6,width_ratios=[4,1.25,4,1.25,4,1.25],wspace=.10,left=.07,right=.99,top=.955,bottom=.10)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']; markers=['o','s','^','D']
    titles=['Exact F1','Group F1','Test log loss']
    for j,endpoint in enumerate(ENDPOINTS):
        ax=fig.add_subplot(grid[0,2*j]); valax=fig.add_subplot(grid[0,2*j+1],sharey=ax)
        ax.axvline(0,color='#666666',linewidth=.7,zorder=0)
        for i,c in enumerate(CONDITIONS):
            for k,r in enumerate(RANKERS):
                v=groups[r,c]['endpoints'][endpoint]; y=i+(k-1.5)*.17
                ax.plot(v['mean'],y,markers[k],color=colors[k],markersize=3)
                ax.hlines(y,*v['ci95'],color=colors[k],linewidth=.85)
                label='0' if v['mean']==0 else f"{v['mean']:.2g}"
                valax.text(.94,y,label,ha='right',va='center',color=colors[k],fontsize=7.2)
            ax.axhline(i+.5,color='#e0e0e0',linewidth=.4,zorder=0)
        ax.set_title(titles[j]);valax.set_title('Mean',fontsize=7)
        ax.set_ylim(7.6,-.6);ax.set_yticks(range(8),CONDITIONS if j==0 else ['']*8)
        ax.tick_params(axis='y',length=3 if j==0 else 0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3));ax.grid(axis='x',color='#dddddd',linewidth=.4)
        ax.spines[['top','right']].set_visible(False)
        valax.set_xlim(0,1);valax.axis('off')
    handles=[Line2D([],[],marker=markers[i],color=colors[i],linestyle='None',label=LABELS[i],markersize=4) for i in range(4)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.016),ncol=4,frameon=False,fontsize=8)
    fig.text(.5,.060,'CPSS minus own-ranker raw reference (unscaled difference)',ha='center',fontsize=8)
    for ext in ['pdf','png']:fig.savefig(OUT/f'fig_paired_quality_intervals.{ext}',dpi=600,metadata={'Creator':OP} if ext=='pdf' else None)
    plt.close(fig)
    (OUT/'numeric_preservation.json').write_text(json.dumps(numeric,indent=2)+'\n',encoding='utf-8')
    # Exact input CSV remains a traceable derivative, never a replacement for
    # the frozen experiment summary.
    (OUT/'condition_ranker_summary.csv').write_bytes((PUB/'condition_ranker_summary.csv').read_bytes())
    record={'created_at':datetime.now().astimezone().isoformat(),'operation_id':OP,'tool':'Codex','model':'gpt-6-astra xhigh','issue_id':'ISS-08',
      'authority':'MD/09_audit_revision/weekend_20260905/AUTHOR_AUTHORITY.md',
      'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [BASE/'analysis/summary.json',PUB/'manifest.json',PUB/'condition_ranker_summary.csv',EXP/'prepare_publication_outputs.py']],
      'script_sha256':sha(Path(__file__)),'matplotlib_version':matplotlib.__version__,
      'scientific_computation':'NONE; saved summary values and intervals only',
      'preservation':{'groups':32,'movement_cells':32,'paired_means':96,'paired_intervals':96,'all_values_equal':True,'all_denominators_equal':True,'original_heatmap_A_labels_equal':True},
      'carrier_reason':'Vector PDF deliberately retained for small line/interval marks and searchable numeric labels; 600 dpi PNG derivative also saved.',
      'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(OUT.iterdir())]}
    (OUT/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'DERIVED_VALUES_VERIFIED','directory':str(OUT),'means':96,'intervals':96}))
if __name__=='__main__':main()
