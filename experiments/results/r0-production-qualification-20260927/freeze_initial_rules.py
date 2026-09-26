"""Reproduce D_dev-only bootstrap initial floors; never reads D_gate rows."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from llm_router_labeling.deployment.r0_release import _evaluation_contract, r0_evaluation_profile
from llm_router_labeling.evaluation.continual_gate import evaluate_continual_predictions

ROOT = Path('/Users/yoocheolmin/workspace/school/LLM_Router')
CODE = ROOT / '.worktrees/r0-ml'
OUT = ROOT / 'ml/artifacts/r0-release-20260927/qualification'
MODEL_VERSION = 'router-modernbert-r0-a-20260927-v1'
WEIGHTS_SHA = 'e1a40fec7d269d8f925b5391e4f5e75df5d0fedbb1e37b4eea96521e27b1a6a7'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def put(name, value):
    path=OUT/name
    if path.exists():
        assert json.loads(path.read_text()) == value, name
        return path
    with path.open('x') as handle: json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False); handle.write('\n')
    return path

paths={
 'original_dev': ROOT/'ml/artifacts/modernbert-initial-v1/dataset/validation/validation.parquet',
 'objective_dev': ROOT/'ml/artifacts/routing-objective-continual-v1/dataset/validation.parquet',
 'scores': ROOT/'experiments/results/context-length-qualification-20260925/r0/scores/validation-8192.parquet',
 'score_manifest': ROOT/'experiments/results/context-length-qualification-20260925/r0/scores/manifest.json',
 'weights': ROOT/'ml/artifacts/r0-release-20260927/source/extracted/final-model/model.safetensors',
}
expected={
 'original_dev':'fb698a992f05b88bfd4488c7de0abe515e52107c8435a46fa9a45bfb63b7f4d0',
 'objective_dev':'aa9d973fcf886c7465fbf9e06701984d099cd74a5eabf508cc610845613edef8',
 'scores':'c655911481462518c01854a587c89c6c91e7d166d3157cfc3053e7e3cc591069',
 'weights': WEIGHTS_SHA,
}
actual_sha={k:sha(p) for k,p in paths.items()}
assert all(actual_sha[k]==v for k,v in expected.items())
manifest=json.loads(paths['score_manifest'].read_text())
assert manifest['model_files']['model.safetensors']==WEIGHTS_SHA
assert manifest['datasets']['validation']['sha256']==expected['objective_dev']
assert manifest['datasets']['validation']['8192']['score_sha256']==expected['scores']
dev=pd.read_parquet(paths['original_dev']); objective=pd.read_parquet(paths['objective_dev']); scores=pd.read_parquet(paths['scores'])
assert len(dev)==len(objective)==len(scores)==3581
for frame in (dev,objective,scores):
    assert frame['sample_id'].notna().all() and frame['sample_id'].is_unique
assert set(dev.sample_id)==set(objective.sample_id)==set(scores.sample_id)
objective=objective.set_index('sample_id').loc[dev.sample_id].reset_index()
scores=scores.set_index('sample_id').loc[dev.sample_id].reset_index()
shared_columns=list(dev.columns)
for col in shared_columns:
    pd.testing.assert_series_equal(dev[col].reset_index(drop=True),objective[col].reset_index(drop=True),check_dtype=False)
logits=scores[['score_low','score_medium','score_high']].to_numpy(dtype=float)
assert np.isfinite(logits).all()
labels=np.array(['easy','medium','hard'])
prediction=pd.DataFrame({'predicted_label':labels[logits.argmax(axis=1)],'was_truncated':scores.was_truncated})
config=r0_evaluation_profile(json.loads((CODE/'ml/config/initial_v1_dev_evaluation.json').read_text()))
gate_config=r0_evaluation_profile(json.loads((CODE/'ml/config/continual_gate_v1.json').read_text()))
# Only the frozen D_gate CONTRACT is read; no D_gate data or scores are accessed.
result=evaluate_continual_predictions(dev,prediction,config,model_info={
    'model_version':MODEL_VERSION,'model_weights_sha256':WEIGHTS_SHA,
    'max_length':8192,'training_max_length':512,'device':manifest['device'],
    'dtype':manifest['dtype'],'attention':manifest['attention'],
    'torch':manifest['torch'],'transformers':manifest['transformers'],
    'prediction_source':'verified-historical-8192-logits',
    'source_scores_sha256':actual_sha['scores'],'source_manifest_sha256':actual_sha['score_manifest'],
})
metrics=result.metrics; rows=result.predictions
put('development.json',config)
put('evaluation.json',gate_config)
metric_path=put('development-metrics.json',metrics)
rows.to_parquet(OUT/'development-predictions.parquet',index=False)

# Cluster bootstrap: sample duplicate_group_id with replacement; retain every row
# in each sampled cluster, including cross-source rows. Same replicates for metrics.
groups=sorted(rows.duplicate_group_id.unique())
group_index=pd.Categorical(rows.duplicate_group_id,categories=groups).codes
truth=pd.Categorical(rows.difficulty_label,categories=labels).codes
pred=pd.Categorical(rows.predicted_label,categories=labels).codes
cell=truth*3+pred
masks={'macro_f1':np.ones(len(rows),dtype=bool),
       **{f'per_source_dataset.{v}.macro_f1':rows.source_dataset.eq(v).to_numpy() for v in sorted(rows.source_dataset.unique())},
       **{f'per_domain.{v}.macro_f1':rows.domain.eq(v).to_numpy() for v in sorted(rows.domain.unique())}}

def metric_map(weights):
    values={}
    for name,mask in masks.items():
        c=np.bincount(cell[mask],weights=weights[mask],minlength=9).reshape(3,3)
        support=c.sum(axis=1); denom=support+c.sum(axis=0)
        f1=np.divide(2*c.diagonal(),denom,out=np.zeros(3),where=denom!=0)
        included=np.ones(3,dtype=bool) if name=='macro_f1' else support>0
        assert included.any()
        values[name]=float(f1[included].mean())
        if name=='macro_f1':
            assert support[2]>0
            values['per_class.hard.recall']=float(c[2,2]/support[2])
    return values

point=metric_map(np.ones(len(rows)))
for name,value in point.items():
    m=metrics
    for part in name.split('.'):m=m[part]
    assert np.isclose(value,m,rtol=0,atol=1e-14),(name,value,m)
rng=np.random.default_rng(42)
draws={k:[] for k in point}
for _ in range(1000):
    counts=np.bincount(rng.integers(0,len(groups),size=len(groups)),minlength=len(groups))
    for name,value in metric_map(counts[group_index]).items():draws[name].append(value)
intervals={k:{'estimate':point[k],'lower':float(np.quantile(v,.025)),'upper':float(np.quantile(v,.975))} for k,v in draws.items()}
floors={k:v['lower'] for k,v in intervals.items()}
small_domains={str(name): {'development_row_count':int(len(group)), 'reason':'insufficient_support', 'quality_unverified':True}
               for name,group in rows.groupby('domain') if len(group)<30}
for name in small_domains: floors.pop(f'per_domain.{name}.macro_f1')
assert all(v>0 for v in floors.values()), 'Unexpected non-positive sufficiently-supported floor'
method={
    'resamples':1000,'seed':42,'resampling_unit':'duplicate_group_id',
    'unique_groups':len(groups),'rows':len(rows),
    'interval':'pointwise two-sided 95% percentile interval; floor is 2.5th percentile',
    'multiple_comparison_adjustment':False,
    'subgroup_macro_policy':'macro_over_labels_with_true_support',
    'note':'Initial label-consistency floors only; not actual service quality or evidence of improvement. D_dev was used during model development and is not an independent test. D_gate floors will not be retuned after observing D_gate.',
}
rules={
 'schema_version':'r0-initial-rules-v1',
 'rules_version':'r0-a-initial-8192-ddev-group-bootstrap-20260927-v1',
 'required_evaluation_contract':_evaluation_contract(gate_config),
 'minimum_metrics':floors,
 'floor_selection_method':method,
 'small_domain_policy':{'policy_version':'initial-small-domain-report-only-v1','minimum_development_rows':30,'frozen_before_gate':True,'scope':'initial_baseline_only'},
 'report_only_domains':small_domains,
}
pilot={
 'schema_version':'r0-initial-pilot-v1','objective':'ordinal',
 'evaluation_role':'development_calibration','dataset_sha256':actual_sha['original_dev'],
 'pilot_metrics_sha256':sha(metric_path),'candidate_model_version':MODEL_VERSION,
 'candidate_weights_sha256':WEIGHTS_SHA,
 'length_contract_version':'train-inference-lengths-v1','inference_max_length':8192,
 'selected_rules':rules.copy(),'bootstrap_intervals':intervals,
 'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'gate_data_read':False,
}
pilot_path=put('initial-pilot.json',pilot)
rules['threshold_selection']={
 'evaluation_role':'development_calibration','dataset_sha256':actual_sha['original_dev'],
 'frozen_before_gate':True,'report_sha256':sha(pilot_path),
}
rules_path=put('initial-rules.json',rules)
audit={
 'schema_version':'r0-initial-ddev-provenance-v1','source_files':{k:{'path':str(paths[k]),'sha256':v} for k,v in actual_sha.items()},
 'rows':len(dev),'sample_id_set_equal':True,'compared_shared_columns':shared_columns,
 'source_scores_join':'unique sample_id exact join reordered to frozen original D_dev',
 'shared_values_equal':True,'gate_data_read':False,'bootstrap_method':method,
 'metrics_sha256':sha(metric_path),'pilot_sha256':sha(pilot_path),'rules_sha256':sha(rules_path),
 'script_sha256':sha(Path(__file__)),
}
put('development-provenance.json',audit)
print(json.dumps({'accuracy':metrics['accuracy'],'intervals':intervals,'rules_sha256':sha(rules_path)},indent=2))
