"""R2/R3 follow-up: frozen competitors, dev-only fitting, and identical G replay."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
sys.path.insert(0, str(ROOT / "src"))
import common_track as C
import gain_eval as G
import lock as L
import ours as O
import r23_extension as R
import test_eval as T
from test_router_final_gain import synthetic_rows, write_scores, write_calibrated

REAL_PREFLIGHT = R.preflight


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    """Use the real calibration, lock, gain and bootstrap; mock only external source loading."""
    run, repo = tmp_path / "run", tmp_path / "repo"
    repo.mkdir()
    exclusion = repo / "excluded.csv"
    pd.DataFrame(columns=["sample_id", "source_dataset", "role"]).to_csv(exclusion, index=False)
    exclusions = {"non_english": {"file": exclusion.name, "sha256": C.sha256_file(exclusion), "rows": 0}}
    rng = np.random.default_rng(142)
    frames = {role: synthetic_rows(rng, role, n=60) for role in ("dev", "test")}
    files = {(name, "default"): write_scores(run / "s" / f"{name}.parquet", frames, noise, rng)
             for name, noise in (("knn", .1), ("r1_group", .15))}
    refs = {name: {"calibrated_file": f"calibrated/{name}.parquet",
                   "calibrated_sha256": write_calibrated(run / "lock/calibrated" / f"{name}.parquet",
                                                         frames, L.A_SOURCES)}
            for name in ("r0_operational", "rt08")}
    old = {"methods": {}, "exclusions": exclusions, "rt08": {"group": {"models": refs}},
           "evaluation": {"sensitivity": {"length": {"lengths_sha256": "L",
               "normalizer_mean_dev_length": {s: 100. for s in L.SOURCES}}}}}
    R.write(run / "lock/selection-lock.json", old)
    outcomes = {"dev": frames["dev"], "train": frames["dev"], "hashes": {},
                "blended": frames["dev"].assign(difficulty_label="easy", **{
                    f"success_{t}": frames["dev"][f"p_{t}"] for t in L.TIERS})}
    monkeypatch.setattr(L, "load_outcomes", lambda _: outcomes)
    monkeypatch.setattr(L, "evaluation_exclusions", lambda _: (set(), exclusions))
    monkeypatch.setattr(L, "score_files", lambda *a: (files, []))
    monkeypatch.setattr(L, "group_prices", lambda _: ({s: np.array([.02, .06, 1.]) for s in L.SOURCES}, "g"))
    ids = pd.concat(frames.values()).sample_id
    length = pd.Series(np.full(len(ids), 100.), index=ids)
    monkeypatch.setattr(L, "length_table", lambda _: (length, {"sha256": "L"}))
    monkeypatch.setitem(T.BOOT, "iterations", 200)
    def truth_loader(*args):
        f = frames["dev" if args[-1] else "test"].copy()
        f["needed"] = f.needed_tier.map({t: i for i, t in enumerate(L.TIERS)})
        f["group_id"] = f.source_dataset + ":" + (f.index // 2).astype(str)
        best = f[[f"p_{t}" for t in L.TIERS]].max(axis=1)
        f["stratum"] = np.where(best >= .8, "solvable", np.where(best > 0, "beyond_partial", "beyond_zero"))
        return {"frame": f, "role": "dev" if args[-1] else "test", "test_sha256": "test",
                "counts": {"evaluated": f.groupby("source_dataset").size().to_dict()},
                "input_sha256": {"groups.json": "g"}}
    monkeypatch.setattr(T, "load_truth", truth_loader)
    G.lock(run, repo, run / "gain-lock", False)
    G.evaluate(run, repo, repo, run / "gain-test", False, 200)
    contract = {"base_gain_lock_sha256": C.sha256_file(run / "gain-lock/gain-lock.json")}
    monkeypatch.setattr(R, "preflight", lambda *a: contract)
    new_scores = {m: write_scores(run / R.EXTENSION / "scores" / m / "scores.parquet", frames, .08, rng)
                  for m in R.METHODS}
    monkeypatch.setattr(R, "verified_scores", lambda run, method, contract: new_scores[method])
    return run, repo, frames, new_scores


def test_append_lock_and_full_gain_reproduce_frozen_competitors(frozen_run):
    run, repo, frames, _ = frozen_run
    original_bytes = (run / "gain-lock/gain-lock.json").read_bytes()
    baseline = R.read(run / "gain-lock/gain-lock.json")
    lock_dir = run / R.EXTENSION / "gain-lock"
    record = R.append_lock(run, repo, lock_dir)
    assert not record["test_outcomes_read"]
    for method, info in baseline["methods"].items():
        assert record["methods"][method] == info
        assert (lock_dir / info["calibrated_file"]).read_bytes() == (run / "gain-lock" / info["calibrated_file"]).read_bytes()
    out = run / R.EXTENSION / "gain-test"
    manifest = R.evaluate(run, repo, repo, out)
    assert manifest["baseline_point_and_interval_reproduced"]
    assert manifest["newly_trained_methods"] == ["r2_group", "r3_group"]
    assert (run / "gain-lock/gain-lock.json").read_bytes() == original_bytes
    macro = pd.read_csv(out / "gain-macro.csv")
    for name in R.METHODS:
        assert set(macro[macro.method.eq(name)].basis) == {"main", "group_price", "length"}
    pairs = pd.read_csv(out / "pairs.csv")
    assert any(set((r.a, r.b)) == {"r2_group", "r3_group"} for r in pairs.itertuples())
    assert (out / "r0-r3-comparison.csv").is_file()
    with pytest.raises(SystemExit, match="exists"):
        R.evaluate(run, repo, repo, out)


def test_lock_does_not_depend_on_test_outcomes(frozen_run):
    run, repo, frames, _ = frozen_run
    first = R.append_lock(run, repo, run / "lock-one")
    for t in L.TIERS:
        frames["test"][f"p_{t}"] = 1 - frames["test"][f"p_{t}"]
    second = R.append_lock(run, repo, run / "lock-two")
    for method in R.METHODS:
        assert first["methods"][method] == second["methods"][method]


def test_mutated_competitor_policy_rejected(frozen_run):
    run, repo, _, _ = frozen_run
    folder = run / R.EXTENSION / "gain-lock"
    record = R.append_lock(run, repo, folder)
    record["methods"]["knn"]["lambda"][L.SOURCES[0]][0]["lambda"] += .1
    R.write(folder / "gain-lock.json", record)
    with pytest.raises(ValueError, match="frozen competitor"):
        R.assert_baseline_unchanged(run, repo, folder)


def test_score_reads_no_test_outcome_columns(tmp_path, monkeypatch):
    prep = tmp_path / "prepare"
    prep.mkdir()
    outputs = {}
    for role in ("dev", "test"):
        path = prep / f"{role}.parquet"
        pd.DataFrame({"sample_id": [role], "source_dataset": ["routerbench"],
                      "model_input_text": ["question"], "p_low": [0.123]}).to_parquet(path)
        outputs[path.name] = {"sha256": C.sha256_file(path)}
    R.write(prep / "manifest.json", {"outputs": outputs})
    read = pd.read_parquet
    def guarded(path, **kwargs):
        assert kwargs["columns"] == ["sample_id", "source_dataset", "model_input_text"]
        return read(path, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    inputs = R.score_inputs(tmp_path)
    assert len(inputs["texts"]) == 2
    assert not any(c.startswith("p_") for c in inputs["scored"])


@pytest.mark.parametrize("method,objective", list(R.METHODS.items()))
def test_training_uses_existing_soft_group_data(tmp_path, monkeypatch, method, objective):
    from llm_router_labeling.objective_comparison import training
    data = tmp_path / "ours/data/blended"
    data.mkdir(parents=True)
    hashes = {}
    for name in ("train.parquet", "validation.parquet"):
        (data / name).write_text("frozen-input")
        hashes[name] = C.sha256_file(data / name)
    R.write(data.parent / "manifest.json", {"variants": {"blended": {"files": hashes}}})
    monkeypatch.setattr(R, "preflight", lambda *a: {})
    seen = {}
    def fake_train(**kwargs):
        seen.update(kwargs)
        kwargs["output_dir"].mkdir()
        return {}
    monkeypatch.setattr(training, "train_objective", fake_train)
    R.train(tmp_path, tmp_path, method, "cpu", tmp_path / "model", smoke=False)
    assert seen["method"] == objective
    assert seen["training_path"] == data / "train.parquet"
    assert seen["seed"] == 42
    assert seen["maximum_train_steps"] == 0
    assert C.sha256_file(data / "train.parquet") == hashes["train.parquet"]


def test_group_losses_have_finite_gradients():
    import torch
    from llm_router_labeling.objective_comparison.objectives import objective_loss
    targets = torch.tensor([[.2, .8, .9], [.9, .8, .7], [0., 0., 0.], [.5, .5, .5]])
    for method in R.METHODS.values():
        scores = torch.zeros((4, 3), requires_grad=True)
        loss = objective_loss(method=method, logits=scores, difficulty_ids=torch.zeros(4, dtype=torch.long),
                              success_targets=targets, quality_targets=targets,
                              decision_costs=torch.tensor([[.2, .6, 1.]] * 4),
                              sample_weights=torch.ones(4),
                              regret_lambdas=[0., .05, .1, .2, .4, .8, 1.6, 3.2], pairwise_tie_weight=.25)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(scores.grad).all()


def test_real_preflight_checks_frozen_training_and_never_reads_test(frozen_run, monkeypatch):
    run, repo, frames, _ = frozen_run
    config = repo / R.CONFIG
    config.parent.mkdir(parents=True)
    config.write_text('{}')
    prep, data = run / 'prepare', run / 'ours/data'
    prep.mkdir()
    (data / 'blended').mkdir(parents=True)
    prep_outputs, data_files = {}, {}
    raw_frames = {'train': synthetic_rows(np.random.default_rng(99), 'train', 70), 'dev': frames['dev']}
    for role, raw in raw_frames.items():
        raw = raw.assign(model_input_text='question', **{f'service_cost_{t}': L.SERVICE_COST[i]
                                                        for i, t in enumerate(L.TIERS)})
        raw.to_parquet(prep / f'{role}.parquet')
        prep_outputs[f'{role}.parquet'] = {'sha256': C.sha256_file(prep / f'{role}.parquet')}
        name = 'train.parquet' if role == 'train' else 'validation.parquet'
        O.to_objective_frame(raw, pd.Series(['test-task'] * len(raw))).to_parquet(data / 'blended' / name)
        data_files[name] = C.sha256_file(data / 'blended' / name)
    R.write(prep / 'manifest.json', {'outputs': prep_outputs})
    R.write(data / 'manifest.json', {'variants': {'blended': {'files': data_files}}})
    R.write(data / 'blended/outcome-manifest.json', {'dataset_version': 'frozen'})
    anchor = run / 'ours/models/r1_group/manifest.json'
    anchor.parent.mkdir(parents=True)
    R.write(anchor, {'experiment_config_sha256': C.sha256_file(config),
                    'training_data_sha256': data_files['train.parquet'],
                    'validation_data_sha256': data_files['validation.parquet'],
                    'outcome_manifest_sha256': C.sha256_file(data / 'blended/outcome-manifest.json'),
                    'seed': 42, 'maximum_train_rows': 0, 'maximum_validation_rows': 0, 'maximum_train_steps': 0})
    baseline = R.read(run / 'gain-lock/gain-lock.json')
    baseline['inputs'] = {'prepare_manifest': C.sha256_file(prep / 'manifest.json'),
                          'ours_data_manifest': C.sha256_file(data / 'manifest.json')}
    R.write(run / 'gain-lock/gain-lock.json', baseline)
    test_manifest = R.read(run / 'gain-test/manifest.json')
    test_manifest['gain_lock_sha256'] = C.sha256_file(run / 'gain-lock/gain-lock.json')
    R.write(run / 'gain-test/manifest.json', test_manifest)
    read = pd.read_parquet
    def guarded(path, *args, **kwargs):
        assert Path(path).name != 'test.parquet'
        return read(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', guarded)
    contract = REAL_PREFLIGHT(run, repo)
    assert not contract['test_outcomes_read']
    assert sum(contract['rows']['train'].values()) == 280
    (data / 'blended/train.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed input'):
        REAL_PREFLIGHT(run, repo)


def test_smoke_model_cannot_enter_evaluation(tmp_path):
    manifest = {'method': 'pairwise', 'seed': 42, 'training_data_sha256': 'train',
                'validation_data_sha256': 'dev', 'experiment_config_sha256': 'config',
                'maximum_train_rows': 64, 'maximum_validation_rows': 64, 'maximum_train_steps': 2}
    R.write(tmp_path / 'manifest.json', manifest)
    (tmp_path / '_SUCCESS').write_text(C.sha256_file(tmp_path / 'manifest.json'))
    contract = {'training_data_sha256': 'train', 'validation_data_sha256': 'dev',
                'training_config_sha256': 'config'}
    with pytest.raises(ValueError, match='smoke model'):
        R.verified_model(tmp_path, 'r2_group', contract)


def test_delivered_notebook_cells_compile():
    for name in ("12a_r2_group", "12b_r3_group", "13_r23_gain"):
        nb = R.read(ROOT / "notebooks/router_final" / f"{name}.ipynb")
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]), filename=f"{name}:cell{i}")
                assert c["outputs"] == [] and c["execution_count"] is None
