"""Post-hoc D_dev diagnostics; preserves fixed-policy action counts per category."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from llm_router_labeling.objective_comparison.evaluation import route_actions
ROOT=Path('/private/tmp/llm-router-objective-results-local/comparison')
METHODS=['ordinal','pointwise','pairwise'];TIERS=['low','medium','high'];SOURCES=['routerbench','xroutebench'];SEED=42;ITERATIONS=1000
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lockpath=ROOT/'validation-lock.json';lock=json.loads(lockpath.read_text())
frames=[];provenance={}
for method in METHODS:
 path=ROOT/method/'validation-scores.parquet';manifest=json.loads((ROOT/method/'manifest.json').read_text());record=manifest['artifacts']['validation-scores.parquet'];assert sha(path)==record['sha256'] and path.stat().st_size==record['size_bytes']
 f=pd.read_parquet(path).sort_values('sample_id').reset_index(drop=True);assert len(f)==3581 and not f.sample_id.duplicated().any();frames.append(f);provenance[method]={'path':str(path),'sha256':record['sha256']}
base=frames[0];reference=[c for c in base if not c.startswith('score_')]
for f in frames[1:]:pd.testing.assert_frame_equal(base[reference],f[reference])
Q=base[[f'quality_{t}' for t in TIERS]].to_numpy(float);C=base[[f'actual_cost_{t}' for t in TIERS]].to_numpy(float);D=base[[f'decision_cost_{t}' for t in TIERS]].to_numpy(float)
assert np.isfinite(Q).all() and np.isfinite(C).all()
source_indices={s:np.flatnonzero(base.source_dataset.eq(s).to_numpy()) for s in SOURCES}
strata=[(str(s),str(c),np.asarray(idx,dtype=np.int64)) for (s,c),idx in base.groupby(['source_dataset','category'],sort=True).groups.items()]
actions=np.empty((len(METHODS),len(base)),dtype=np.int64);penalties={}
for mi,method in enumerate(METHODS):
 scores=frames[mi][[f'score_{t}' for t in TIERS]].to_numpy(float);penalties[method]={}
 for source,idx in source_indices.items():
  penalty=float(lock['methods'][method]['locked_budget_penalties'][source]['0.5000']);penalties[method][source]=penalty;actions[mi,idx]=route_actions(method,scores[idx],D[idx],penalty)
observed=np.empty((3,2,2),dtype=float)
for si,source in enumerate(SOURCES):
 idx=source_indices[source]
 for mi in range(3):observed[mi,si]=[Q[idx,actions[mi,idx]].mean(),C[idx,actions[mi,idx]].mean()]
null=np.empty((ITERATIONS,3,2,2),dtype=float);rng=np.random.default_rng(SEED)
stratum_counts={(mi,s,c):np.bincount(actions[mi,idx],minlength=3) for s,c,idx in strata for mi in range(3)}
for it in range(ITERATIONS):
 shuffled=np.empty_like(actions)
 for s,c,idx in strata:
  permutation=rng.permutation(idx)
  shuffled[:,idx]=actions[:,permutation]
  for mi in range(3):assert np.array_equal(np.bincount(shuffled[mi,idx],minlength=3),stratum_counts[(mi,s,c)])
 for si,source in enumerate(SOURCES):
  idx=source_indices[source]
  for mi in range(3):null[it,mi,si]=[Q[idx,shuffled[mi,idx]].mean(),C[idx,shuffled[mi,idx]].mean()]
def summarize(actual,values):
 lo,hi=np.quantile(values,[.025,.975]);mean=values.mean()
 return {'actual':float(actual),'shuffle_mean':float(mean),'shuffle_std':float(values.std(ddof=1)),'shuffle_p025':float(lo),'shuffle_p975':float(hi),'actual_minus_shuffle_mean':float(actual-mean),'actual_vs_central_95pct_range':'above' if actual>hi else 'below' if actual<lo else 'within','descriptive_fraction_shuffles_at_least_actual':float(np.mean(values>=actual))}
rows=[]
for mi,method in enumerate(METHODS):
 for si,source in enumerate(SOURCES):
  idx=source_indices[source]
  rows.append({'method':method,'source':source,'rows':len(idx),'frozen_lambda':penalties[method][source],'action_counts':dict(zip(TIERS,np.bincount(actions[mi,idx],minlength=3).tolist())),'mean_quality':summarize(observed[mi,si,0],null[:,mi,si,0]),'mean_actual_cost':summarize(observed[mi,si,1],null[:,mi,si,1]),'null_quality_cost_correlation':float(np.corrcoef(null[:,mi,si,0],null[:,mi,si,1])[0,1])})
report={'schema_version':'within_source_category_action_permutation_diagnostic_v1','role':'posthoc_D_dev_descriptive_diagnostic_not_held_out_hypothesis_test','budget':.5,'seed':SEED,'iterations':ITERATIONS,'rng':'numpy.random.default_rng (PCG64)','numpy_version':np.__version__,'stratification':['source_dataset','category'],'operation':'Permute frozen selected action labels among rows within each source/category. Each iteration uses the same row permutations for all 3 methods. Outcomes, costs, questions and frozen lambda remain unchanged.','preservation':'Exact low/medium/high counts preserved for every method/source/category in every iteration; asserted. Row-specific selected actual cost need not be preserved.','range_meaning':'Empirical 2.5th to 97.5th percentiles of random reassignment outcomes conditional on fixed dev rows and action counts; not a confidence interval, training-seed interval, causal estimate, or held-out test.','limit':'Questions were used to select objectives and lambda before this post-hoc diagnostic. Source/category invariance and cost equality are not assumed. Do not infer equal-cost superiority from quality differences.','metric_units':'Quality and actual cost use each source benchmark original outcome/cost units; no source costs pooled or interpreted as current Claude spend.','validation_lock':{'path':str(lockpath),'sha256':sha(lockpath),'winner_unchanged':lock['winner_method']},'artifacts':provenance,'strata':[{'source':s,'category':c,'rows':len(idx),'action_counts':{method:dict(zip(TIERS,stratum_counts[(mi,s,c)].tolist())) for mi,method in enumerate(METHODS)}} for s,c,idx in strata],'results':rows,'null_draws':{'axis_order':['iteration','method','source','metric'],'methods':METHODS,'sources':SOURCES,'metrics':['mean_quality','mean_actual_cost'],'values':null.tolist()},'all_count_preservation_assertions_passed':True,'model_training_or_inference_performed':False,'policy_reselection_performed':False}
output=Path('/private/tmp/llm-router-dev-within-category-permutation.json');output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
lines=['# D_dev: 출처·카테고리 내부 선택 셔플 진단','', '고정 budget 0.50의 선택 행동을 같은 source/category 안에서만 1,000회(seed 42) 재배정했다. 모든 반복에서 각 방법·출처·카테고리의 low/medium/high 사용 건수가 정확히 보존됨을 확인했다. 세 방법에는 같은 행 permutation을 적용했다. 모델 추론·학습·정책 재선택은 없다.','', '| 방법 | source | 실제 품질 | 셔플 평균 [p2.5, p97.5] | 실제−평균 | 실제 비용 | 셔플 비용 평균 [p2.5, p97.5] |','|---|---|---:|---|---:|---:|---|']
for row in rows:
 q=row['mean_quality'];c=row['mean_actual_cost'];lines.append(f'| {row["method"]} | {row["source"]} | {q["actual"]:.6f} | {q["shuffle_mean"]:.6f} [{q["shuffle_p025"]:.6f}, {q["shuffle_p975"]:.6f}] | {q["actual_minus_shuffle_mean"]:+.6f} | {c["actual"]:.9f} | {c["shuffle_mean"]:.9f} [{c["shuffle_p025"]:.9f}, {c["shuffle_p975"]:.9f}] |')
lines+=['', '이는 이미 선택에 사용한 D_dev의 사후 진단이다. 대괄호는 같은 표본·선택 건수에 조건부인 무작위 재배정 결과의 중앙 95% 범위이며 신뢰구간이 아니다. 통계적 유의성 검정, held-out 성능, causal 효과를 주장하지 않는다.','', '선택 비율이 같아도 문항별 실제 비용이 달라 총비용은 보존되지 않는다. 품질이 셔플보다 높은 경우에도 동일비용 우월로 해석할 수 없다. 비용은 각 benchmark의 원래 단위이며 현재 Claude 비용이나 서비스 절감률이 아니다.','', 'JSON에는 18개 source/category stratum의 행동 건수, 1,000회 원시 집계, 파일 SHA, 고정 λ, source별 품질·비용 범위를 보관했다.']
Path('/private/tmp/llm-router-dev-within-category-permutation.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'output':str(output),'results':rows},ensure_ascii=False,indent=2,allow_nan=False))
