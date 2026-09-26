import sys,json,importlib.util,argparse
from pathlib import Path
import numpy as np,pandas as pd
helper=Path(__file__).parents[1]/'r1/analyze_length.py';spec=importlib.util.spec_from_file_location('helper',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
p=argparse.ArgumentParser();p.add_argument('--scores-dir',type=Path,required=True);a=p.parse_args()
assert (a.scores_dir/'_SUCCESS').read_text().strip()==h.sha(a.scores_dir/'manifest.json'), 'Incomplete or changed inference manifest'
manifest=json.loads((a.scores_dir/'manifest.json').read_text())
expected=json.loads((Path(__file__).parents[1]/'inputs.json').read_text())
for file in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','training_config.json']:
 assert manifest['model_files'][file]==expected['models/ordinal/'+file]['sha256'], file
for dataset,record in manifest['datasets'].items():
 assert record['sha256']==expected['data/'+dataset+'.parquet']['sha256'], dataset
 for length in [512,8192]:assert record[str(length)]['score_sha256']==h.sha(a.scores_dir/f'{dataset}-{length}.parquet')
root=Path('/Users/yoocheolmin/workspace/school/LLM_Router'); audit=json.loads((helper.parent/'dataset-audit.json').read_text());results={}
for name,key in [('validation','a_validation'),('test_routerbench','a_test_routerbench'),('test_xroutebench','a_test_xroutebench'),('arena','arena')]:
 if key not in audit: print('Available keys',list(audit));raise SystemExit(1)
 rows=pd.read_parquet(audit[key]['path']).reset_index(drop=True);s512,n=h.aligned_scores(rows,a.scores_dir/f'{name}-512.parquet');s8192,n2=h.aligned_scores(rows,a.scores_dir/f'{name}-8192.parquet');assert np.array_equal(n,n2)
 x=s512.argmax(1);z=s8192.argmax(1);masks={'all':np.ones(len(rows),bool),'over512':n>512,'at_most512':n<=512,'over8192':n>8192}
 for col in ['source_dataset','category','evaluation_split','arena_source_dataset']:
  if col in rows:
   for value in sorted(rows[col].astype(str).unique()):
    m=rows[col].astype(str).eq(value).to_numpy();masks[f'{col}={value}']=m;masks[f'{col}={value}/over512']=m&(n>512)
 result={'rows':len(rows),'decision':'v1 softmax argmax (no cost penalty)','arena_combined_all_descriptive_only':name=='arena','primary_arena_slice':'evaluation_split=full' if name=='arena' else None,'dataset_sha256':h.sha(audit[key]['path']),'scores_sha256':{str(k):h.sha(a.scores_dir/f'{name}-{k}.parquet') for k in [512,8192]},'slices':{}}
 for label,m in masks.items():
  if not m.any():continue
  r=rows.loc[m].reset_index(drop=True)
  if 'source_sample_id' not in r:r['source_sample_id']=r['duplicate_group_id'] if 'duplicate_group_id' in r else r['sample_id']
  xx=x[m];zz=z[m];v=h.measure(r,xx,zz,True)
  y=r.difficulty_label.map(dict(zip(h.LABELS,range(3)))).to_numpy(); hard=y==2
  for length,action in [(512,xx),(8192,zz)]:v['lengths'][str(length)]['hard_recall']=float((action[hard]==2).mean()) if hard.any() else None
  result['slices'][label]=v
 result['short_rawlogit_max_abs_difference']=float(np.abs(s512[n<=512]-s8192[n<=512]).max())
 if name=='arena':
  old=pd.read_parquet(root/'ml/artifacts/routerarena-diagnostics-v1-20260924/report/predictions.parquet');old=old[old.method=='ordinal'].set_index('sample_id').loc[rows.sample_id];o=old[h.SCORE_COLS].to_numpy(float);result['baseline_reproduction']={'rows':len(old),'changed_actions':int((o.argmax(1)!=x).sum()),'max_abs_logit_difference':float(np.abs(o-s512).max())}
 results[name]=result
(a.scores_dir/'r0-comparison.json').write_text(json.dumps(results,indent=2))
print(json.dumps({k:{'rows':v['rows'],'delta':v['slices']['all']['primary_quality_delta']} for k,v in results.items()}))
