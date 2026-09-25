"""Visualize stored post-hoc D_dev reassignment draws; no new experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/llm-router-arena-visuals-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def render(args):
    source=args.input_prefix
    data=json.loads(source.with_suffix('.json').read_text())
    if data['role']!='posthoc_D_dev_descriptive_diagnostic_not_held_out_hypothesis_test':
        raise ValueError('Expected post-hoc dev diagnostic')
    for artifact in [data['validation_lock'],*data['artifacts'].values()]:
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Input artifact checksum mismatch')
    raw=data['null_draws']
    methods=raw['methods'];sources=raw['sources'];metrics=raw['metrics']
    draws=np.asarray(raw['values'],dtype=float)
    if raw['axis_order'] != ['iteration','method','source','metric'] or draws.shape!=(data['iterations'],3,2,2) or not np.isfinite(draws).all():
        raise ValueError('Invalid stored null draw shape')
    records=[]
    for result in data['results']:
        mi=methods.index(result['method']);si=sources.index(result['source'])
        if sum(result['action_counts'].values()) != result['rows']:
            raise ValueError('Action count support mismatch')
        for ki,metric in enumerate(metrics):
            stored=result[metric];values=draws[:,mi,si,ki]
            expected=dict(shuffle_mean=float(values.mean()),shuffle_p025=float(np.quantile(values,.025)),shuffle_p975=float(np.quantile(values,.975)))
            if any(not np.isclose(stored[k],v,rtol=1e-12,atol=1e-15) for k,v in expected.items()):
                raise ValueError('Stored null range mismatch')
            records.append({'method':result['method'],'source':result['source'],'metric':metric,'n':result['rows'],**stored})
    output=args.output_dir
    if output.exists():
        raise ValueError('Output must not exist')
    output.mkdir(parents=True)
    for suffix in ['.json','.md','.py']:
        shutil.copyfile(source.with_suffix(suffix),output/('original-audit'+suffix))
    shutil.copyfile(__file__,output/'render.py')
    pd.DataFrame(records).to_csv(output/'ranges.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#172b45','axes.titleweight':'bold'})
    fig,axes=plt.subplots(2,2,figsize=(14,7.8))
    fig.subplots_adjust(top=.80,bottom=.15,left=.17,right=.97,hspace=.64,wspace=.25)
    for si,source_name in enumerate(sources):
        for ki,metric in enumerate(metrics):
            ax=axes[si,ki]
            for mi,method in enumerate(methods):
                stored=next(r for r in records if r['source']==source_name and r['method']==method and r['metric']==metric)
                center=stored['shuffle_mean']
                lo=stored['shuffle_p025']-center;hi=stored['shuffle_p975']-center;actual=stored['actual']-center
                ax.plot([lo,hi],[mi,mi],color='#a8bacd',linewidth=9,solid_capstyle='butt',label='Central 95% reassignment range (not CI)' if mi==0 else None)
                ax.plot(0,mi,'|',color='#52637a',markersize=15)
                ax.plot(actual,mi,'D',color=['#63738a','#087f8c','#8b5fbf'][mi],markersize=7,label='Observed frozen policy' if mi==0 else None)
            ax.axvline(0,color='#9cadbf',linewidth=.8,linestyle='--')
            ax.set(yticks=range(3),yticklabels=['R0','R1','R2'],ylim=(2.6,-.6),title=f'{source_name} · '+('quality' if metric=='mean_quality' else 'actual cost'),xlabel='Δ from shuffle mean'+(' (source cost units)' if ki==1 else ' (quality points)'))
            ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
            if ki==1:
                ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3),useMathText=True)
    fig.suptitle('Post-hoc D_dev · within-category action reassignment',x=.02,y=.97,ha='left',fontsize=17,fontweight='bold')
    fig.text(.02,.9,'Same source/category action counts · 1,000 paired shuffles · budget 0.5 frozen · quality and cost both change.',fontsize=10,color='#52637a')
    fig.text(.02,.855,'Ranges describe random reassignments, not confidence intervals or held-out significance. Cost units are source-local, not current service spend.',fontsize=9.5,color='#52637a')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.02))
    fig.savefig(output/'posthoc-quality-cost-ranges.png',dpi=150,bbox_inches='tight');plt.close(fig)
    verification={'schema_version':'dev-category-permutation-visuals-v1','source_input_sha256':sha(source.with_suffix('.json')),'artifact_hashes_verified':True,'stored_null_draw_means_and_quantiles_recomputed':True,'new_inference':False,'new_permutation':False,'range_is_confidence_interval':False,'costs_are_source_local':True}
    (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({'schema_version':'dev-category-permutation-artifacts-v1','files':{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}},indent=2)+'\n')
    print(json.dumps({'output':str(output),'shape':list(draws.shape),'figures':1,'verification':verification}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-prefix',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    render(parser.parse_args())
