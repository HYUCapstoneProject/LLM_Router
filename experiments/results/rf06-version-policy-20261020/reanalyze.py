"""Audit RF-06 saved predictions and receipts; never train models or modify Drive.

Usage: .venv/bin/python experiments/results/rf06-version-policy-20261020/reanalyze.py \
    --run-dir /tmp/rf06-review-run001 --data-dir ml/artifacts/version-policy-v1/data
The run directory is a read-only input snapshot without model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cm(y, pred):
    return np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)


def f1(matrix):
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    den = matrix.sum(-1) + matrix.sum(-2)
    return np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--policy-code', type=Path, default=Path('.worktrees/rf06/ml/notebooks/version_policy'))
    args = p.parse_args()
    run, data, out = args.run_dir, args.data_dir, Path(__file__).resolve().parent
    plan = json.loads((run / 'plan/plan.json').read_text())
    original = json.loads((run / 'compare/results.json').read_text())
    assert sha(run / 'plan/plan.json') == (run / 'plan/_SUCCESS').read_text().strip()
    assert sha(data / 'manifest.json') == plan['data_manifest_sha256']
    verified = 0
    for name, digest in plan['files'].items():
        assert sha(run / 'plan' / name) == digest, name
        verified += 1
    for name, entry in json.loads((data / 'manifest.json').read_text())['outputs'].items():
        expected = entry if isinstance(entry, str) else entry['sha256']
        assert sha(data / name) == expected, name
        verified += 1
    archive = run / 'code/version-policy-source.zip'
    assert sha(archive) == archive.with_suffix('.sha256').read_text().split()[0]
    with zipfile.ZipFile(archive) as z:
        for name in ['plan.py', 'rounds.py', 'compare.py', 'config.json']:
            assert z.read('ml/notebooks/version_policy/' + name) == (args.policy_code / name).read_bytes()
    sys.path.insert(0, str(args.policy_code.resolve()))
    import plan as policy
    frames = {s: pd.read_parquet(data / (s + '.parquet')) for s in ['pool', 'dev', 'test']}
    sealed = {s: pd.read_parquet(run / 'plan' / (s + '.parquet')) for s in ['seed', 'batches', 'probe']}
    receipts = {}
    training_checks = []
    for arm in ['p1', 'p2']:
        rr = []
        for k in range(1, 10):
            d = run / 'arms' / arm / 'seed42' / f'round{k:02d}'
            assert (d / '_SUCCESS').exists()
            r = json.loads((d / 'receipt.json').read_text())
            assert r['plan_sha256'] == sha(run / 'plan/plan.json')
            for name, digest in r['outputs'].items():
                assert sha(d / name) == digest
                verified += 1
            anchor = ((k - 1) // 3) * 3 if arm == 'p1' else 0
            expected_hash = rr[anchor - 1]['model_sha256'] if anchor else plan['m0_model_sha256']
            assert r['initial_model_sha256'] == expected_hash
            assert r['anchor_round'] == anchor
            expected = (policy.p1_rows(k, sealed['batches'], sealed['seed'], plan['config']['version_policy'])
                        if arm == 'p1' else policy.p2_rows(k, sealed['batches'], sealed['seed']))
            actual = pd.read_parquet(d / 'training-rows.parquet')
            assert not actual.sample_id.duplicated().any()
            assert set(zip(actual.sample_id, actual.part)) == set(zip(expected.sample_id, expected.part))
            assert r['train_rows'] == len(actual)
            assert r['optimizer_steps'] == 3 * int(np.ceil(len(actual) / 32))
            training_checks.append({'arm': arm, 'round': k, 'anchor_round': anchor, 'rows': len(actual), 'verified': True})
            rr.append(r)
        receipts[arm] = rr

    overlap = {}
    for a, b in [('pool', 'dev'), ('pool', 'test'), ('dev', 'test')]:
        overlap[a + '-' + b] = {col: len(set(frames[a][col]) & set(frames[b][col])) for col in ['sample_id', 'group_id']}
        norm = lambda s: s.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        overlap[a + '-' + b]['exact_normalized_text'] = len(set(norm(frames[a].model_input_text)) & set(norm(frames[b].model_input_text)))
    test = frames['test']
    y = test.difficulty_label.map({'easy': 0, 'medium': 1, 'hard': 2}).to_numpy()
    paths = {'p0': run / 'plan/m0-test-scores.parquet', **{a: run / 'arms' / a / 'seed42/round09/test-scores.parquet' for a in receipts}}
    predictions = {}
    for arm, path in paths.items():
        d = pd.read_parquet(path)
        assert not d.sample_id.duplicated().any() and set(d.sample_id) == set(test.sample_id)
        predictions[arm] = d.set_index('sample_id').loc[test.sample_id, ['score_low', 'score_medium', 'score_high']].to_numpy().argmax(1)
    metrics, per_source = {}, []
    sources = sorted(test.source_dataset.unique())
    for arm, pred in predictions.items():
        conf = cm(y, pred)
        scores = []
        for source in sources:
            mask = test.source_dataset.eq(source).to_numpy()
            score = float(f1(cm(y[mask], pred[mask])).mean())
            scores.append(score)
            per_source.append({'arm': arm, 'source': source, 'n': int(mask.sum()), 'macro_f1': score})
        expected = next(r for r in original['final_test'] if r['arm'] == arm)
        assert np.isclose(np.mean(scores), expected['test_source_macro_f1'], atol=1e-12)
        metrics[arm] = {'source_macro_f1': float(np.mean(scores)), 'accuracy': float((pred == y).mean()),
                        'class_f1': dict(zip(['easy', 'medium', 'hard'], f1(conf).tolist())),
                        'class_recall': dict(zip(['easy', 'medium', 'hard'], (np.diag(conf) / conf.sum(1)).tolist())),
                        'predicted_share': dict(zip(['easy', 'medium', 'hard'], (np.bincount(pred, minlength=3) / len(pred)).tolist())),
                        'confusion_rows_true_columns_pred': conf.tolist()}

    # Recompute paired source-stratified group bootstrap from confusion contributions.
    rng = np.random.default_rng(42)
    draws = {arm: np.zeros(2000) for arm in predictions}
    for source in sources:
        mask = test.source_dataset.eq(source).to_numpy()
        groups, _ = pd.factorize(test.loc[mask, 'group_id'])
        n = groups.max() + 1
        sample = rng.multinomial(n, np.full(n, 1 / n), size=2000)
        for arm, pred in predictions.items():
            counts = np.zeros((n, 9))
            np.add.at(counts, (groups, y[mask] * 3 + pred[mask]), 1)
            draws[arm] += f1((sample @ counts).reshape(2000, 3, 3)).mean(-1) / len(sources)
    intervals = {}
    for a, b in [('p1', 'p2'), ('p1', 'p0'), ('p2', 'p0')]:
        ci = np.quantile(draws[a] - draws[b], [0.025, 0.975]).tolist()
        assert np.allclose(ci, original['verdicts'][a + '-' + b]['ci'], atol=1e-12)
        intervals[a + '-' + b] = ci

    compute = {}
    for arm, rr in receipts.items():
        compute[arm] = {key: sum(r[key] for r in rr) for key in ['optimizer_steps', 'fit_seconds', 'score_seconds', 'planned_samples']}
        compute[arm]['tokens'] = sum(r['tokens_per_epoch'] * r['epochs'] for r in rr)
    compute['fit_saving_pct'] = 100 * (1 - compute['p1']['fit_seconds'] / compute['p2']['fit_seconds'])
    compute['step_saving_pct'] = 100 * (1 - compute['p1']['optimizer_steps'] / compute['p2']['optimizer_steps'])
    compute['pilot_fit_seconds'] = sum(r['fit_seconds'] for r in plan['pilot_receipts'].values())
    rounds = pd.read_csv(run / 'compare/tables/per-round.csv')
    preq = pd.read_csv(run / 'compare/tables/prequential.csv')
    preq_mean = preq.groupby('arm').f1.mean().to_dict()
    checks = {'verified_file_hashes': verified, 'code_archive_sha256': sha(archive),
              'plan_sha256': sha(run / 'plan/plan.json'), 'split_overlap': overlap,
              'training_receipts': training_checks, 'model_weights_rehashed': False,
              'weight_chain_check': 'Receipt SHA chain only; model weights were not downloaded.',
              'source': 'Google Drive MyDrive/LLM_Router/version-policy/runs/run001',
              'seed': 42, 'bootstrap_intervals_reproduced': intervals, 'metrics': metrics,
              'compute': compute, 'prequential_window_mean': preq_mean,
              'actual_total_unique_training_rows': len(sealed['seed']) + len(sealed['batches']),
              'pilot_largest_training_rows': 16000}
    (out / 'audit.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
    pd.DataFrame(per_source).to_csv(out / 'test-by-source.csv', index=False)
    for filename in ['results.json', 'results-for-review.zip']:
        shutil.copyfile(run / 'compare' / filename, out / ('original-' + filename))
    for filename in ['per-round.csv', 'final-test.csv', 'prequential.csv']:
        shutil.copyfile(run / 'compare/tables' / filename, out / filename)
    shutil.copyfile(run / 'plan/plan.json', out / 'sealed-plan.json')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 12,
                         'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.9), constrained_layout=True)
    for ax in axs:
        ax.grid(axis='y', alpha=.16)
        ax.set_xticks(range(1, 10))
        ax.set_xlabel('재학습 라운드')
    labels = {'p1': 'P1 버전 정책', 'p2': 'P2 전체 재학습'}
    colors = {'p1': '#087F8C', 'p2': '#C6563E'}
    for arm in ['p1', 'p2']:
        d = rounds[rounds.arm.eq(arm)]
        axs[0].plot(d['round'], d.dev, 'o-', color=colors[arm], linewidth=2.2, label=labels[arm])
        axs[1].plot(d['round'], d.fit_seconds / 60, 'o-', color=colors[arm], linewidth=2.2, label=labels[arm])
    axs[0].axhline(plan['pilot_curve']['1000'], color='#8F969A', linestyle='--', label='P0 동결')
    axs[0].set_title('초반 P2 우세, 후반 P1 추월', loc='left', weight='bold', pad=15)
    axs[0].set_ylabel('dev source-macro-F1')
    axs[0].set_ylim(.315, .435)
    axs[0].legend(loc='lower right', frameon=False)
    axs[1].set_title('9회 갱신의 학습 시간 64.0% 절감', loc='left', weight='bold', pad=15)
    axs[1].set_ylabel('누적 학습 시간 (분)')
    axs[1].set_ylim(0, 114)
    axs[1].annotate('P2 101.6분', (9, compute['p2']['fit_seconds'] / 60), xytext=(-86, 9), textcoords='offset points', color=colors['p2'], weight='bold')
    axs[1].annotate('P1 36.6분', (9, compute['p1']['fit_seconds'] / 60), xytext=(-80, 12), textcoords='offset points', color=colors['p1'], weight='bold')
    fig.suptitle('RF-06 · seed 42 · Tesla T4 · 9라운드', x=.02, ha='left', fontsize=17, weight='bold')
    fig.savefig(out / 'quality-and-training-cost.png', dpi=180)
    plt.close(fig)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
