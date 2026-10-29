"""Append R2/R3 group routers to the frozen EV-10 G comparison.

Only these two routers are trained. The original blended rows, source weights,
group success targets, 1:3:5 prices and trainer are reused. Calibration fits dev
only; the original gain lock and calibrated competitors are preserved bytewise.
Outputs live under <run>/r23-extension, never in the original result folders.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import common_track as C
import gain_eval as G
import lock as L
import ours as O

METHODS = {"r2_group": "pairwise", "r3_group": "regret"}
EXTENSION = "r23-extension"
CONFIG = "ml/config/routing_objective_continual_v1.json"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n")


def check(path, digest):
    if not Path(path).is_file() or C.sha256_file(Path(path)) != digest:
        raise ValueError(f"missing or changed input: {path}")


def preflight(run_dir: Path, repo: Path) -> dict:
    """Check the frozen baseline and train/dev data without reading test outcomes."""
    base = G.load_gain_lock(run_dir, repo)
    rec = base["record"]
    if set(METHODS) & set(rec["methods"]):
        raise ValueError("the base gain lock already contains the extension")
    check(run_dir / "lock/selection-lock.json", rec["base_lock_sha256"])
    for key, relative in (("prepare_manifest", "prepare/manifest.json"),
                          ("ours_data_manifest", "ours/data/manifest.json")):
        check(run_dir / relative, rec["inputs"][key])
    np.testing.assert_allclose(rec["evaluation"]["service_cost"], L.SERVICE_COST, rtol=0, atol=0)
    if rec["evaluation"]["tau"] != G.TAU:
        raise ValueError("base gain lock uses another tau")
    # Anchor the actual training data and configuration to the completed R1 run.
    anchor = read(run_dir / "ours/models/r1_group/manifest.json")
    check(repo / CONFIG, anchor["experiment_config_sha256"])
    check(run_dir / "ours/data/blended/train.parquet", anchor["training_data_sha256"])
    check(run_dir / "ours/data/blended/validation.parquet", anchor["validation_data_sha256"])
    check(run_dir / "ours/data/blended/outcome-manifest.json", anchor["outcome_manifest_sha256"])
    if anchor["seed"] != 42 or any(anchor[k] for k in
            ("maximum_train_rows", "maximum_validation_rows", "maximum_train_steps")):
        raise ValueError("the anchor must be the complete seed-42 R1 run")
    data_manifest = read(run_dir / "ours/data/manifest.json")
    prep = read(run_dir / "prepare/manifest.json")
    for filename in ("train.parquet", "dev.parquet"):
        check(run_dir / "prepare" / filename, prep["outputs"][filename]["sha256"])
    rows, ids = {}, {}
    for split in ("train", "validation"):
        path = run_dir / "ours/data/blended" / f"{split}.parquet"
        check(path, data_manifest["variants"]["blended"]["files"][path.name])
        df = pd.read_parquet(path)
        if not df.sample_id.is_unique or set(df.source_dataset) != set(L.SOURCES):
            raise ValueError(f"{split}: duplicate ids or missing source")
        ids[split] = set(df.sample_id)
        p = df[[f"quality_{t}" for t in L.TIERS]].to_numpy(float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"{split}: invalid group targets")
        np.testing.assert_allclose(p, df[[f"success_{t}" for t in L.TIERS]], rtol=0, atol=0)
        for prefix in ("decision_cost", "actual_cost"):
            np.testing.assert_allclose(df[[f"{prefix}_{t}" for t in L.TIERS]],
                                       np.broadcast_to(L.SERVICE_COST, p.shape), rtol=0, atol=1e-12)
        np.testing.assert_allclose(df.sample_weight, O.balanced_weight(df), rtol=1e-10, atol=1e-12)
        rows[split] = df.groupby("source_dataset").size().to_dict()
    if ids["train"] & ids["validation"]:
        raise ValueError("train/dev overlap")
    _, groups_sha = L.group_prices(run_dir)
    # groups.json belongs to the same frozen preparation as the existing G test.
    old_test = read(run_dir / "gain-test/manifest.json")
    if old_test["input_sha256"]["groups.json"] != groups_sha:
        raise ValueError("model groups differ from the completed G comparison")
    if old_test["gain_lock_sha256"] != base["sha256"]:
        raise ValueError("baseline test was produced with another gain lock")
    return {"base_gain_lock_sha256": base["sha256"], "groups_sha256": groups_sha,
            "training_config_sha256": C.sha256_file(repo / CONFIG),
            "training_data_sha256": anchor["training_data_sha256"],
            "validation_data_sha256": anchor["validation_data_sha256"],
            "rows": rows, "seed": 42, "test_outcomes_read": False,
            "target": "group success probability; quality_* = success_*",
            "cost": L.SERVICE_COST.tolist(), "methods": METHODS}


def verified_model(folder: Path, method: str, contract: dict):
    manifest = read(folder / "manifest.json")
    check(folder / "manifest.json", (folder / "_SUCCESS").read_text().strip())
    if manifest["method"] != METHODS[method] or manifest["seed"] != 42:
        raise ValueError(f"wrong model identity: {method}")
    for key in ("training_data_sha256", "validation_data_sha256"):
        if manifest[key] != contract[key]:
            raise ValueError(f"{method}: {key} differs")
    if manifest["experiment_config_sha256"] != contract["training_config_sha256"]:
        raise ValueError(f"{method}: training config differs")
    if any(manifest[k] for k in ("maximum_train_rows", "maximum_validation_rows", "maximum_train_steps")):
        raise ValueError("smoke model cannot enter the full comparison")
    for rel, info in manifest["model_artifacts"].items():
        check(folder / "model" / rel, info["sha256"])
    for rel, info in manifest["artifacts"].items():
        check(folder / rel, info["sha256"])
    return manifest


def train(run_dir: Path, repo: Path, method: str, device: str, output_dir: Path, smoke: bool):
    contract = preflight(run_dir, repo)
    if (output_dir / "_SUCCESS").exists() and not smoke:
        verified_model(output_dir, method, contract)
        print(f"{method}: verified completed model; training skipped", flush=True)
        return
    result = O.train(repo, run_dir / "ours/data", method, output_dir, smoke, device, seed=42)
    write(output_dir / "extension-contract.json", contract | {"smoke": smoke})
    return result


def publish(run_dir: Path, repo: Path, method: str, local_model: Path):
    contract = preflight(run_dir, repo)
    verified_model(local_model, method, contract)
    target = run_dir / EXTENSION / "models" / method
    if target.exists():
        verified_model(target, method, contract)
        if C.sha256_file(target / "manifest.json") != C.sha256_file(local_model / "manifest.json"):
            raise ValueError("a different completed model already exists in Drive")
        return
    pending = target.with_name(target.name + ".uploading")
    if pending.exists():
        raise ValueError(f"incomplete earlier upload; inspect or move {pending}")
    pending.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_model, pending)
    verified_model(pending, method, contract)
    pending.rename(target)


def score_inputs(run_dir: Path) -> dict:
    """Read only identifiers and text for dev/test; outcomes stay unopened."""
    prep = read(run_dir / "prepare/manifest.json")
    parts = []
    hashes = {}
    for role in ("dev", "test"):
        path = run_dir / "prepare" / f"{role}.parquet"
        hashes[path.name] = prep["outputs"][path.name]["sha256"]
        check(path, hashes[path.name])
        rows = pd.read_parquet(path, columns=["sample_id", "source_dataset", "model_input_text"])
        parts.append(rows.assign(role=role))
    frame = pd.concat(parts, ignore_index=True)
    if not frame.sample_id.is_unique:
        raise ValueError("duplicate question ids across dev/test")
    return {"scored": frame[["sample_id", "source_dataset", "role"]],
            "texts": frame.model_input_text.astype(str).tolist(), "hashes": hashes}


def score(run_dir: Path, repo: Path, method: str, device: str):
    import score_modernbert as S
    contract = preflight(run_dir, repo)
    model_folder = run_dir / EXTENSION / "models" / method
    model = verified_model(model_folder, method, contract)
    inputs = score_inputs(run_dir)
    output = run_dir / EXTENSION / "scores"
    folder = output / method
    if folder.exists():
        return verified_scores(run_dir, method, contract)
    models = {method: (str((model_folder / "model").relative_to(run_dir)), L.TIERS)}
    result = S.score(method, run_dir, inputs, output, device, models=models)
    write(folder / "extension-score-manifest.json", {
        "method": method, "inputs": inputs["hashes"], "test_outcomes_read": False,
        "training_manifest_sha256": C.sha256_file(model_folder / "manifest.json"),
        "training_data_sha256": model["training_data_sha256"],
        "scores_sha256": result["scores_sha256"], "contract": contract})
    return result


def verified_scores(run_dir: Path, method: str, contract: dict) -> Path:
    model_folder = run_dir / EXTENSION / "models" / method
    verified_model(model_folder, method, contract)
    folder = run_dir / EXTENSION / "scores" / method
    if not (folder / "_SUCCESS").exists():
        raise ValueError(f"incomplete scoring output: {folder}")
    record = read(folder / "extension-score-manifest.json")
    if record["contract"] != contract or record["test_outcomes_read"] or record["method"] != method:
        raise ValueError(f"scoring contract differs: {method}")
    check(model_folder / "manifest.json", record["training_manifest_sha256"])
    for name, digest in record["inputs"].items():
        check(run_dir / "prepare" / name, digest)
    check(folder / "scores.parquet", record["scores_sha256"])
    return folder / "scores.parquet"


def append_lock(run_dir: Path, repo: Path, output_dir: Path):
    contract = preflight(run_dir, repo)
    base = G.load_gain_lock(run_dir, repo)
    rec = copy.deepcopy(base["record"])
    data = L.load_outcomes(run_dir)
    dev = data["dev"][~data["dev"].sample_id.isin(base["excluded"])].reset_index(drop=True)
    cols = [f"p_{t}" for t in L.TIERS]
    new_calibrated, selection = {}, []
    for method in METHODS:
        path = verified_scores(run_dir, method, contract)
        scores = pd.read_parquet(path)
        if scores.duplicated(["sample_id", "role"]).any():
            raise ValueError(f"duplicate score rows: {method}")
        scores = scores[~scores.sample_id.isin(base["excluded"])].reset_index(drop=True)
        if not np.isfinite(scores[[f"score_{t}" for t in L.TIERS]]).all().all():
            raise ValueError(f"non-finite scores: {method}")
        oof, frozen = L.calibrate_method(method, scores, dev, cols, {}, L.SOURCES)
        new_calibrated[method] = frozen
        lam, metrics = {}, {}
        for source, (_, _, Y, Phat) in oof.items():
            targets = G.first_cost_targets(Y)
            np.testing.assert_allclose(targets, rec["evaluation"]["cost_levels"]["targets"][source])
            metrics[source] = G.gain_point(Y, L.decide("tau", Phat, 0.0, tau=G.TAU))
            lam[source] = G.lambda_first(Y, Phat, targets)
            selection.append({"method": method, "source": source, **metrics[source]})
        rec["methods"][method] = {
            "selected_setting": "default", "selected_setting_07b": None, "changed_from_07b": False,
            "new_extension_method": True, "settings_tried": 1, "kind": "per_tier",
            "dev_mean_G": float(np.mean([v["G"] for v in metrics.values()])),
            "lambda": lam, "scores_file": str(path.relative_to(run_dir)),
            "scores_sha256": C.sha256_file(path), "calibrated_file": f"calibrated/{method}.parquet"}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "calibrated").mkdir()
    # Copy only the frozen competitors; never refit/reselect them to favour the additions.
    for method, info in base["record"]["methods"].items():
        dest = output_dir / info["calibrated_file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base["files"][method], dest)
        check(dest, info["calibrated_sha256"])
    for method, frame in new_calibrated.items():
        path = output_dir / rec["methods"][method]["calibrated_file"]
        frame.to_parquet(path, index=False)
        rec["methods"][method]["calibrated_sha256"] = C.sha256_file(path)
    old_table = run_dir / "gain-lock/selection-table.parquet"
    check(old_table, rec["selection_table_sha256"])
    shutil.copyfile(old_table, output_dir / old_table.name)
    pd.DataFrame(selection).to_csv(output_dir / "r23-dev.csv", index=False)
    rec["experiment"] = "EV-10 R2/R3 model-group extension, frozen baseline G"
    rec["extension"] = contract | {"original_methods_unchanged": list(base["record"]["methods"]),
        "selection": "one fixed final checkpoint per new method; only adapters and lambda fitted on dev",
        "test_reuse": "follow-up comparison on the existing test; not a new unseen test",
        "dev_table_sha256": C.sha256_file(output_dir / "r23-dev.csv")}
    write(output_dir / "gain-lock.json", rec)
    summary = ["# R2·R3 집단 비교 — dev 고정 완료", "",
               "기존 라우터의 설정·보정·λ는 그대로 복사했다. R2·R3만 dev로 보정하고 λ를 정했다.",
               "test 결과는 읽지 않았다. 주 비용은 1:3:5이며 집단 가격·길이 비용은 τ=.8 민감도로 평가한다.",
               "", "| 방법 | dev G (네 원천 평균) |", "|---|---:|"]
    summary += [f"| {m} | {rec['methods'][m]['dev_mean_G']:+.5f} |" for m in METHODS]
    (output_dir / "gain-lock-summary.md").write_text("\n".join(summary) + "\n")
    return rec


def assert_baseline_unchanged(run_dir: Path, repo: Path, lock_dir: Path):
    original = G.load_gain_lock(run_dir, repo)
    current = G.load_gain_lock(run_dir, repo, lock_dir)
    record = current["record"]
    if record["extension"]["base_gain_lock_sha256"] != original["sha256"]:
        raise ValueError("base gain lock changed after the extension was locked")
    if set(record["methods"]) != set(original["record"]["methods"]) | set(METHODS):
        raise ValueError("unexpected methods in extension lock")
    for method, info in original["record"]["methods"].items():
        if record["methods"][method] != info:
            raise ValueError(f"frozen competitor changed: {method}")
    for key in ("evaluation", "references", "seeds", "exclusions", "inputs", "base_lock_sha256"):
        if record[key] != original["record"][key]:
            raise ValueError(f"frozen baseline contract changed: {key}")
    contract = preflight(run_dir, repo)
    for method in METHODS:
        path = verified_scores(run_dir, method, contract)
        check(path, record["methods"][method]["scores_sha256"])
    return original


def compare_baseline(old: pd.DataFrame, new: pd.DataFrame):
    keys = ["basis", "source", "method", "rule", "param"]
    metrics = ["A", "C", "H", "G", "start_low", "start_medium", "start_high", "G_lo", "G_hi"]
    current = new[~new.method.isin(METHODS)].set_index(keys).sort_index()
    before = old.set_index(keys).sort_index()
    if not before.index.equals(current.index):
        raise ValueError("baseline evaluation row set changed")
    np.testing.assert_allclose(current[metrics], before[metrics], rtol=1e-8, atol=1e-9, equal_nan=True)


def evaluate(run_dir: Path, repo: Path, data_root: Path, output_dir: Path, rehearse=False):
    lock_dir = run_dir / EXTENSION / "gain-lock"
    original = assert_baseline_unchanged(run_dir, repo, lock_dir)
    # Full test always retains the baseline's 10,000 paired bootstrap draws.
    manifest = G.evaluate(run_dir, repo, data_root, output_dir, rehearse,
                          1000 if rehearse else G.T.BOOT["iterations"], gain_lock_dir=lock_dir)
    if not rehearse:
        baseline = read(run_dir / "gain-test/manifest.json")
        check(run_dir / "gain-test/gain.parquet", baseline["outputs"]["gain.parquet"])
        compare_baseline(pd.read_parquet(run_dir / "gain-test/gain.parquet"),
                         pd.read_parquet(output_dir / "gain.parquet"))
        manifest["baseline_point_and_interval_reproduced"] = True
    manifest["experiment"] = "EV-10 R2/R3 group extension G"
    manifest["base_gain_lock_sha256"] = original["sha256"]
    manifest["newly_trained_methods"] = list(METHODS)
    manifest["test_is_reused_followup"] = not rehearse
    write(output_dir / "manifest.json", manifest)
    for name in ("gain", "gain-macro", "pairs", "by-stratum", "fixed-starts"):
        pd.read_parquet(output_dir / f"{name}.parquet").to_csv(output_dir / f"{name}.csv", index=False)
    macro = pd.read_parquet(output_dir / "gain-macro.parquet")
    focus = macro[macro.method.isin(["r0_group", "r1_group", *METHODS]) & macro.scope.eq("macro")
                  & macro.rule.eq("tau") & np.isclose(macro.param, .8)]
    focus.to_csv(output_dir / "r0-r3-comparison.csv", index=False)
    summary = output_dir / "gain-summary.md"
    summary.write_text("# R2·R3 모델 집단 후속 비교\n\n"
        "R2·R3만 새로 학습했다. 기존 비교군은 같은 선택·보정·λ를 유지한다. "
        "기존 test를 재사용한 후속 비교이며 새로운 독립 test가 아니다. "
        "G는 첫 시도 이득으로 에스컬레이션 효용 U와 구분한다.\n\n" + summary.read_text())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train", "publish", "score", "lock", "test", "rehearse"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-model", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run, repo = args.run_dir, args.repo
    if args.command in ("train", "publish", "score") and args.method is None:
        parser.error("--method is required")
    if args.command in ("train", "rehearse") and args.output_dir is None:
        parser.error("--output-dir is required")
    if args.command == "preflight":
        result = preflight(run, repo)
    elif args.command == "train":
        result = train(run, repo, args.method, args.device, args.output_dir, args.smoke)
    elif args.command == "publish":
        if args.local_model is None:
            parser.error("--local-model is required")
        result = publish(run, repo, args.method, args.local_model)
    elif args.command == "score":
        result = score(run, repo, args.method, args.device)
    elif args.command == "lock":
        result = append_lock(run, repo, run / EXTENSION / "gain-lock")
    else:
        root = args.data_root or run / "inputs/test-data"
        result = evaluate(run, repo, root, args.output_dir or run / EXTENSION / "gain-test",
                          rehearse=args.command == "rehearse")
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
