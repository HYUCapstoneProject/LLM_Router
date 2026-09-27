"""Build reproducible user-run inputs from the existing sealed local artifacts."""
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
from experiment import sha, read, file_pin, validate_frames, exact_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-repo', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    repo, out = a.source_repo.resolve(), a.output_dir.resolve()
    if out.exists():
        raise ValueError('Use a new output directory')
    train = repo / 'ml/artifacts/routing-objective-continual-v1/dataset'
    held = repo / 'ml/artifacts/frozen-evaluation-outcomes/continual-v1-20260924-28cbc342c510'
    arena = repo / 'ml/artifacts/routerarena-diagnostics-v1-20260924/prepared'
    sources = {'train': train/'train.parquet', 'dev': train/'validation.parquet',
               'feedback': held/'stream.parquet', 'gate': held/'gate.parquet',
               'forget': held/'retention_test.parquet', 'routerarena': arena/'evaluation.parquet'}
    frames, provenance = {}, {}
    for role, path in sources.items():
        manifest = read(path.parent/'manifest.json')
        info = manifest.get('artifacts', manifest.get('files'))[path.name]
        file_pin(path, info['sha256'])
        frame = pd.read_parquet(path)
        if role == 'routerarena':
            frame = frame[frame.evaluation_split.eq('full')].copy()
            assert len(frame) == 8379  # Existing frozen full set; robustness stays separate.
        frames[role] = frame.reset_index(drop=True)
        provenance[role] = {'source': str(path.relative_to(repo)), 'source_sha256': sha(path),
                            'filter': 'evaluation_split == full' if role == 'routerarena' else None,
                            'rows': len(frame), 'groups': frame.duplicate_group_id.nunique()}
    validate_frames(frames)
    (out/'data').mkdir(parents=True)
    for role, frame in frames.items():
        frame.to_parquet(out/'data'/f'{role}.parquet', index=False)
    original = repo/'ml/artifacts/modernbert-initial-v1/dataset'
    cohort = {'training_rows_sha256': sha(original/'train/training.parquet'),
              'split_manifest_sha256': sha(original/'manifest.json'),
              'label_definition_sha256': sha(original/'threshold/selected-t1-t2.json'),
              'candidate_pool_sha256': sha(train/'arms.json'),
              'backbone_revision': '8949b909ec900327062f0ebf497f51aef5e6f0c8',
              'initial_optimizer_steps': 2007, 'initial_seed': 42, 'training_max_length': 512}
    models = {}
    for arm, name in [('r0', 'ordinal'), ('r1', 'pointwise')]:
        manifest = read(arena/f'{name}-run-manifest.json')
        assert manifest['common_training_config']['model_revision'] == cohort['backbone_revision']
        assert manifest['seed'] == cohort['initial_seed']
        assert manifest['outcome_manifest_sha256'] == sha(train/'manifest.json')
        models[arm] = {'path': f'models/{arm}', 'files': {k:v['sha256'] for k,v in manifest['model_artifacts'].items()}, 'cohort': cohort}
    lock = read(arena/'validation-lock.json')
    policy = {'sources': {}}
    for source, group in frames['train'].groupby('source_dataset'):
        costs = []
        for tier in ('low','medium','high'):
            values = group[f'decision_cost_{tier}'].unique()
            assert len(values) == 1
            costs.append(float(values[0]))
        policy['sources'][source] = {'lambda': lock['methods']['pointwise']['locked_budget_penalties'][source]['0.5000'], 'decision_costs': costs}
    exact_json(out/'r1-policy.json', policy)
    shutil.copyfile(repo/'ml/config/modernbert_runtime_cu124_v1.json',out/'runtime.json')
    exact_json(out/'source_pins.json', {'schema_version':'router-transfer-source-pins-v1','models':models,
       'data_sha256':{r:sha(out/'data'/f'{r}.parquet') for r in frames},
       'r1_policy':{'path':'r1-policy.json','sha256':sha(out/'r1-policy.json')},
       'runtime_contract':{'path':'runtime.json','sha256':sha(out/'runtime.json')}})
    exact_json(out/'model-archives.json', {
       'r0':{'drive_relative_path':'LLM_Router/modernbert-initial-v1-20260924/attempt-recovery-eb016ac/final-model.tar','sha256':'d15da837971f5535f1f96dcbe036717d066e339afad923d449c8a74e5a041e4f'},
       'r1':{'drive_relative_path':'LLM_Router/routing-objectives-continual-v1-A-20260924/pointwise-final-results.tar','sha256':'0da9d3dc3b845642bf0b8201758308a4d21a3c66fd7582468b46a0ea328a7d8d'}})
    exact_json(out/'data-manifest.json', {'roles':provenance,
        'training_rows_sha256_semantics':'Shared initial question/label Parquet, not identical R0/R1 target-file bytes',
        'frozen_policy_manifest_sha256':sha(arena/'validation-lock.json'),
        'new_domain_adaptation':'future_work_excluded_from_graduation_project'})
    print(json.dumps({r:len(f) for r,f in frames.items()},indent=2))


if __name__ == '__main__':
    main()
