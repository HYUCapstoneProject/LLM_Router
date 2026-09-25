from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score
ROOT=Path('/Users/yoocheolmin/workspace/school/LLM_Router')
TRAIN=ROOT/'ml/artifacts/routing-objective-continual-v1/dataset/train.parquet'
DEV=ROOT/'ml/artifacts/routing-objective-continual-v1/dataset/validation.parquet'
SCORES=Path('/private/tmp/llm-router-objective-results-local/comparison/ordinal/validation-scores.parquet')
LABELS=['easy','medium','hard']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def majority(frame):
 counts=frame.difficulty_label.value_counts().reindex(LABELS,fill_value=0)
 return {'prediction':str(counts.idxmax()),'count':int(counts.sum()),'label_counts':{k:int(v) for k,v in counts.items()},'tied_labels':[x for x in LABELS if counts[x]==counts.max()]}
def metrics(true,pred):
 present=[x for x in LABELS if np.any(true==x)]
 return {'n':len(true),'accuracy':float(accuracy_score(true,pred)),'macro_f1_fixed_3_labels':float(f1_score(true,pred,labels=LABELS,average='macro',zero_division=0)),'balanced_accuracy_present_true_labels':float(recall_score(true,pred,labels=present,average='macro',zero_division=0)),'macro_recall_fixed_3_labels':float(recall_score(true,pred,labels=LABELS,average='macro',zero_division=0)),'true_label_counts':{x:int(np.sum(true==x)) for x in LABELS},'prediction_counts':{x:int(np.sum(pred==x)) for x in LABELS},'confusion_matrix_true_by_pred':confusion_matrix(true,pred,labels=LABELS).tolist()}
train=pd.read_parquet(TRAIN);dev=pd.read_parquet(DEV);score=pd.read_parquet(SCORES)
assert len(train)==21406 and len(dev)==len(score)==3581
assert not dev.sample_id.duplicated().any() and not score.sample_id.duplicated().any()
assert set(dev.sample_id)==set(score.sample_id)
f=dev[['sample_id','source_dataset','category','difficulty_label']].merge(score[['sample_id','source_dataset','category','difficulty_label','score_low','score_medium','score_high']],on='sample_id',validate='one_to_one',suffixes=('','_score'))
for c in ['source_dataset','category','difficulty_label']:assert f[c].equals(f[c+'_score'])
global_prior=majority(train)
source_priors={str(k):majority(g) for k,g in train.groupby('source_dataset',sort=True)}
category_priors={(str(s),str(c)):majority(g) for (s,c),g in train.groupby(['source_dataset','category'],sort=True)}
f['source_majority']=f.source_dataset.map(lambda s:source_priors.get(s,global_prior)['prediction'])
def category_prediction(row):
 if (row.source_dataset,row.category) in category_priors:return category_priors[(row.source_dataset,row.category)]['prediction'],'source_category'
 if row.source_dataset in source_priors:return source_priors[row.source_dataset]['prediction'],'source_fallback'
 return global_prior['prediction'],'global_fallback'
a=f.apply(category_prediction,axis=1);f['source_category_majority']=[x[0] for x in a];f['category_baseline_lookup']=[x[1] for x in a]
f['r0_raw_argmax']=np.asarray(LABELS)[f[['score_low','score_medium','score_high']].to_numpy(float).argmax(axis=1)]
methods=['source_majority','source_category_majority','r0_raw_argmax']
result={'schema_version':'r0_train_only_category_baselines_v1','scope':'D_dev_difficulty_classification_only_R0_no_R1_R2_or_test','label_order':LABELS,'fit_data':'train_only','majority_tie_order':LABELS,'source_fallback':'unseen source -> train global majority','source_category_fallback':'unseen (source,category) -> train source majority; unseen source -> train global majority','metric_definitions':{'macro_f1_fixed_3_labels':'Always average easy/medium/hard F1; absent labels contribute 0. Category rows with fewer than 3 true classes should be read with the class-support counts.','balanced_accuracy_present_true_labels':'Average per-class recall over true labels present in the evaluated slice; equivalent to sklearn balanced_accuracy_score without adjusted=True.','macro_recall_fixed_3_labels':'Average recall across all three label IDs; absent labels contribute 0.'},'artifacts':{k:{'path':str(p),'sha256':sha(p)} for k,p in [('train',TRAIN),('dev',DEV),('r0_raw_scores',SCORES)]},'train_global_prior':global_prior,'train_source_priors':source_priors,'train_source_category_priors':[{'source':s,'category':c,**v} for (s,c),v in category_priors.items()],'lookup_counts':f.category_baseline_lookup.value_counts().to_dict(),'source_fallback_rows':int((~f.source_dataset.isin(source_priors)).sum()),'pooled':{},'by_source':{},'by_source_category':[],'comparison':{}}
for scope,d in [('pooled',f),*list(f.groupby('source_dataset',sort=True))]:
 true=d.difficulty_label.to_numpy();v={m:metrics(true,d[m].to_numpy()) for m in methods}
 if scope=='pooled':result['pooled']=v
 else:result['by_source'][scope]=v
 base=d.source_category_majority.to_numpy();r0=d.r0_raw_argmax.to_numpy();result['comparison'][scope]={'n':len(d),'prediction_agreement':float(np.mean(base==r0)),'both_correct':int(np.sum((base==true)&(r0==true))),'baseline_only_correct':int(np.sum((base==true)&(r0!=true))),'r0_only_correct':int(np.sum((base!=true)&(r0==true))),'both_wrong':int(np.sum((base!=true)&(r0!=true))),'r0_minus_category_baseline_accuracy':v['r0_raw_argmax']['accuracy']-v['source_category_majority']['accuracy'],'r0_minus_category_baseline_macro_f1':v['r0_raw_argmax']['macro_f1_fixed_3_labels']-v['source_category_majority']['macro_f1_fixed_3_labels']}
for (source,category),d in f.groupby(['source_dataset','category'],sort=True):
 true=d.difficulty_label.to_numpy();v={m:metrics(true,d[m].to_numpy()) for m in methods};result['by_source_category'].append({'source':source,'category':category,'train_prior':category_priors.get((source,category)),**v,'r0_minus_category_baseline_accuracy':v['r0_raw_argmax']['accuracy']-v['source_category_majority']['accuracy']})
result['source_macro']={m:{metric:float(np.mean([v[m][metric] for v in result['by_source'].values()])) for metric in ['accuracy','macro_f1_fixed_3_labels','balanced_accuracy_present_true_labels']} for m in methods}
Path('/private/tmp/llm-router-r0-category-baselines.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
f.drop(columns=[c for c in f if c.endswith('_score')]).to_parquet('/private/tmp/llm-router-r0-category-baselines.parquet',index=False)
for scope,v in [('pooled',result['pooled']),*result['by_source'].items(),('source_macro',result['source_macro'])]:
 print(scope)
 for m,x in v.items():print(m,{k:round(x[k],6) for k in ['accuracy','macro_f1_fixed_3_labels','balanced_accuracy_present_true_labels']})
print('comparison',json.dumps(result['comparison']));print('lookup',result['lookup_counts'],'sourcefallback',result['source_fallback_rows'])
for v in result['by_source_category']:
 print('category',v['source'],v['category'],v['r0_raw_argmax']['n'],'trainmajority',v['train_prior']['prediction'],'baselineacc',round(v['source_category_majority']['accuracy'],6),'r0acc',round(v['r0_raw_argmax']['accuracy'],6),'delta',round(v['r0_minus_category_baseline_accuracy'],6))
