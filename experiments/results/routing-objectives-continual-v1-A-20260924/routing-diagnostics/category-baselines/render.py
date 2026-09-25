"""Render and independently check the train-only R0 category baseline audit."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/llm-router-arena-visuals-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

LABELS = ['easy', 'medium', 'hard']
METHODS = ['source_majority', 'source_category_majority', 'r0_raw_argmax']
COLORS = ['#8c9bb0', '#d99b3d', '#087f8c']
NAMES = ['Train source majority', 'Train source + category majority', 'R0 raw argmax']

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def majority(frame):
    return frame.difficulty_label.value_counts().reindex(LABELS, fill_value=0).idxmax()

def validate(data, rows):
    if data['schema_version'] != 'r0_train_only_category_baselines_v1' or data['fit_data'] != 'train_only':
        raise ValueError('Expected train-only baseline audit')
    for artifact in data['artifacts'].values():
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Baseline input checksum mismatch')
    train = pd.read_parquet(data['artifacts']['train']['path'])
    dev = pd.read_parquet(data['artifacts']['dev']['path'])
    scores = pd.read_parquet(data['artifacts']['r0_raw_scores']['path'])
    if set(train.sample_id) & set(dev.sample_id):
        raise ValueError('Train/dev sample overlap')
    reference = dev.set_index('sample_id').sort_index()
    rows = rows.set_index('sample_id').sort_index()
    scores = scores.set_index('sample_id').sort_index()
    if rows.index.has_duplicates or scores.index.has_duplicates or not rows.index.equals(reference.index) or not rows.index.equals(scores.index):
        raise ValueError('Baseline prediction coverage mismatch')
    for key in ['source_dataset','category','difficulty_label']:
        if not rows[key].equals(reference[key]) or not rows[key].equals(scores[key]):
            raise ValueError('Baseline metadata mismatch')
    by_source = train.groupby('source_dataset').apply(majority, include_groups=False).to_dict()
    by_category = train.groupby(['source_dataset','category']).apply(majority, include_groups=False).to_dict()
    global_label = majority(train)
    expected_source = rows.source_dataset.map(lambda s: by_source.get(s, global_label))
    expected_category = rows.apply(lambda r: by_category.get((r.source_dataset,r.category),by_source.get(r.source_dataset,global_label)), axis=1)
    if not rows.source_majority.equals(expected_source) or not rows.source_category_majority.equals(expected_category):
        raise ValueError('Stored majority predictions were not fitted from train only')
    expected_r0 = np.asarray(LABELS)[scores[['score_low','score_medium','score_high']].to_numpy().argmax(1)]
    if not np.array_equal(expected_r0,rows.r0_raw_argmax):
        raise ValueError('R0 raw argmax mismatch')
    stats = []
    for scope, part in [('pooled',rows), *rows.groupby('source_dataset')]:
        stored = data['pooled'] if scope == 'pooled' else data['by_source'][scope]
        truth = part.difficulty_label
        present = [label for label in LABELS if truth.eq(label).any()]
        for method in METHODS:
            pred = part[method]
            values = {'accuracy':float(truth.eq(pred).mean()),
                'macro_f1_fixed_3_labels':float(f1_score(truth,pred,labels=LABELS,average='macro',zero_division=0)),
                'balanced_accuracy_present_true_labels':float(recall_score(truth,pred,labels=present,average='macro',zero_division=0))}
            if stored[method]['n'] != len(part) or not np.array_equal(stored[method]['confusion_matrix_true_by_pred'],confusion_matrix(truth,pred,labels=LABELS)):
                raise ValueError('Audit confusion/support mismatch')
            for metric, value in values.items():
                if not np.isclose(value, stored[method][metric], rtol=1e-12,atol=1e-12):
                    raise ValueError('Audit metric mismatch')
                stats.append({'scope':scope,'method':method,'metric':metric,'n':len(part),'value':value})
    return rows,pd.DataFrame(stats)

def render(args):
    source = args.input_prefix
    data = json.loads(source.with_suffix('.json').read_text())
    rows,stats = validate(data,pd.read_parquet(source.with_suffix('.parquet')))
    output = args.output_dir
    if output.exists():
        raise ValueError('Output must not exist')
    output.mkdir(parents=True)
    for suffix in ['.json','.md','.parquet','.py']:
        shutil.copyfile(source.with_suffix(suffix),output/('original-audit'+suffix))
    shutil.copyfile(__file__,output/'render.py')
    stats.to_csv(output/'metrics.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#172b45','axes.titleweight':'bold'})
    fig,axes=plt.subplots(1,3,figsize=(15,5.5))
    fig.subplots_adjust(top=.77,bottom=.22,left=.14,right=.98,wspace=.18)
    scopes=['pooled','routerbench','xroutebench']
    for index,(ax,metric,title) in enumerate(zip(axes,['accuracy','macro_f1_fixed_3_labels','balanced_accuracy_present_true_labels'],['Accuracy','Macro F1 · fixed 3 labels','Balanced accuracy · present labels'])):
        for i,(method,color) in enumerate(zip(METHODS,COLORS)):
            subset=stats.loc[stats.metric.eq(metric)&stats.method.eq(method)].set_index('scope').reindex(scopes)
            y=np.arange(3)+(i-1)*.24
            ax.barh(y,subset.value,height=.21,color=color,label=NAMES[i])
            for position,value in zip(y,subset.value):
                ax.text(value+.009,position,f'{value:.1%}',va='center',fontsize=9)
        ax.set(yticks=range(3),yticklabels=['Pooled · n=3,581','RouterBench · n=3,122','xRouteBench · n=459'] if index==0 else [],title=title,xlim=(0,.85))
        ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False)
    fig.suptitle('R0 difficulty · train-only source / category baselines',x=.02,y=.97,ha='left',fontsize=17,fontweight='bold')
    fig.text(.02,.885,'Frozen train label majorities → D_dev · R0 raw argmax, λ=0 · no R1/R2, no test, no new inference.',fontsize=10,color='#52637a')
    fig.savefig(output/'r0-train-category-baselines.png',dpi=150,bbox_inches='tight');plt.close(fig)
    # Category differences carry sample counts; a zero does not establish causal shortcut use.
    table = pd.DataFrame([{'source':item['source'],'category':item['category'],'n':item['r0_raw_argmax']['n'],'r0_minus_category_accuracy':item['r0_minus_category_baseline_accuracy']} for item in data['by_source_category']])
    table.to_csv(output/'category-differences.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,6.5))
    fig.subplots_adjust(top=.8,bottom=.13,left=.19,right=.98,wspace=.7)
    for ax,source_name in zip(axes,['routerbench','xroutebench']):
        subset=table.loc[table.source.eq(source_name)].sort_values('category')
        values=subset.r0_minus_category_accuracy.to_numpy()*100
        ax.barh(range(len(subset)),values,color=np.where(values>=0,'#087f8c','#c77970'))
        ax.axvline(0,color='#8190a3',linewidth=1)
        ax.set(yticks=range(len(subset)),yticklabels=[f'{row.category} · n={row.n}' for row in subset.itertuples()],xlabel='R0 − train category majority (percentage points)',title=source_name)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        low=min(-3,float(values.min())-5);high=max(3,float(values.max())+6);ax.set_xlim(low,high)
        for i,value in enumerate(values):
            ax.text(value+(.5 if value>=0 else -.5),i,f'{value:+.2f}',ha='left' if value>=0 else 'right',va='center',fontsize=9)
    fig.suptitle('R0 improvement over a train-only category majority',x=.02,y=.97,ha='left',fontsize=17,fontweight='bold')
    fig.text(.02,.9,'Descriptive D_dev comparison · source/category label priors can explain part of accuracy; this is not a causal shortcut test.',fontsize=9.7,color='#52637a')
    fig.savefig(output/'r0-category-differences.png',dpi=150,bbox_inches='tight');plt.close(fig)
    checks={'schema_version':'r0-category-baseline-visuals-v1','new_inference':False,'train_prior_reconstruction':True,'dev_ids_and_metadata_verified':True,'r0_raw_argmax_verified':True,'pooled_and_source_metrics_recomputed':True,'source_input_sha256':{k:v['sha256'] for k,v in data['artifacts'].items()},'original_audit_sha256':sha(output/'original-audit.json')}
    (output/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    manifest={'schema_version':'r0-category-baseline-artifacts-v1','files':{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(output),'rows':len(rows),'figures':2,'checks':checks}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-prefix',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    render(parser.parse_args())
