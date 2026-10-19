"""RF-06 retraining rounds on the sealed plan (Colab T4).

Arms (plan 2):
  p1     version policy: start from the last .0 anchor (M0 = 1.0), train on the current cycle's batches in full +
         compact history replay (32 per label) + D_seed rehearsal (32 per label). Fresh optimizer every round.
  p2     full retraining: start from M0 every round, train on D_seed and every batch so far.
  p2_eq  (optional) P2's rows, stopped at P1's optimizer steps for the same seed and round.
Every arm trains 3 epochs on its own rows (routing_objective_continual_v1.json), same seed for every round.

After round k the new model scores dev (monitoring), probe window k+1 (prequential: the model in service when that
window arrives) and, at the last round, test (texts only). Each round writes arms/<arm>/seed<N>/round<kk>/ with a
receipt and _SUCCESS; finished rounds are skipped, so a disconnected session resumes at the next round. Model
weights are kept on Drive only where needed: P1 anchors (x.0) and every arm's last round.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

import plan as P

ARMS = ("p1", "p2", "p2_eq")


def round_dir(run_dir: Path, arm: str, seed: int, k: int) -> Path:
    return run_dir / "arms" / arm / f"seed{seed}" / f"round{k:02d}"


def training_rows(arm: str, k: int, sealed: dict, policy: dict) -> pd.DataFrame:
    if arm == "p1":
        return P.p1_rows(k, sealed["batches"], sealed["seed"], policy)
    return P.p2_rows(k, sealed["batches"], sealed["seed"])


def initial_model(arm: str, k: int, run_dir: Path, sealed: dict, seed: int, policy: dict) -> tuple:
    if arm != "p1":
        return sealed["dir"] / "m0-model", 0
    a = P.anchor_round(k, policy["cycle_size"])
    if a == 0:
        return sealed["dir"] / "m0-model", 0
    path = round_dir(run_dir, "p1", seed, a) / "model"
    if not (path / "model.safetensors").exists():
        raise SystemExit(f"P1 anchor {P.version_name(a, policy['cycle_size'])} (round {a}) is missing: {path}")
    return path, a


def step_cap(arm: str, k: int, run_dir: Path, seed: int) -> int:
    if arm != "p2_eq":
        return 0
    receipt = round_dir(run_dir, "p1", seed, k) / "receipt.json"
    if not receipt.exists():
        raise SystemExit(f"p2_eq needs P1 round {k} for seed {seed} first")
    return int(json.loads(receipt.read_text())["optimizer_steps"])


def run_round(repo: Path, cfg: dict, data_dir: Path, out_root: Path, sealed: dict, arm: str, seed: int, k: int,
              device: str, smoke: bool) -> dict:
    policy, rounds = cfg["version_policy"], sealed["plan"]["config"]["stream"]["rounds"]
    target = round_dir(out_root, arm, seed, k)
    if (target / "_SUCCESS").exists():
        print(f"[{arm} seed{seed} round{k}] done", flush=True)
        return json.loads((target / "receipt.json").read_text())
    rows = training_rows(arm, k, sealed, policy)
    init, anchor = initial_model(arm, k, out_root, sealed, seed, policy)
    cap = step_cap(arm, k, out_root, seed)
    local = Path("/content/rf06-work" if Path("/content").exists() else out_root / "_work") / f"{arm}-s{seed}-r{k:02d}"
    print(f"[{arm} seed{seed} round{k}] version {P.version_name(k, policy['cycle_size'])}, start "
          f"{'M0' if anchor == 0 else f'round {anchor}'}, rows {len(rows)} {rows.part.value_counts().to_dict()}"
          f"{f', step cap {cap}' if cap else ''}", flush=True)
    receipt = P.train_model(repo, cfg, rows, sealed["validation"], data_dir / "outcome-manifest.json", local, seed,
                            device, initial_model_dir=init, maximum_train_steps=cap, smoke=smoke)
    data = P.load_data(data_dir)
    dev = data["dev"] if not smoke else data["dev"].groupby("source_dataset").head(40)
    started = time.time()
    outputs = {"dev-scores.parquet": P.score(local / "model", dev, device)}
    probe = sealed["probe"]
    if k < rounds:
        outputs["probe-scores.parquet"] = P.score(local / "model", probe[probe.window.eq(k + 1)], device)
    if k == rounds:
        test = data["test"] if not smoke else data["test"].groupby("source_dataset").head(40)
        outputs["test-scores.parquet"] = P.score(local / "model", test, device)
    receipt.update({"arm": arm, "seed": seed, "round": k, "version": P.version_name(k, policy["cycle_size"]),
                    "anchor_round": anchor, "step_cap": cap, "score_seconds": time.time() - started,
                    "rows_by_part": rows.part.value_counts().to_dict(), "label_mix": P.label_ratio(rows),
                    "rows_by_source": rows.source_dataset.value_counts().to_dict(),
                    "plan_sha256": P.sha256_file(sealed["dir"] / "plan.json"), "smoke": smoke})
    tmp = target.parent / f"{target.name}-tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name, frame in outputs.items():
        frame.to_parquet(tmp / name, index=False)
    rows[["sample_id", "part"]].to_parquet(tmp / "training-rows.parquet", index=False)
    keep_model = (arm == "p1" and k % policy["cycle_size"] == 0) or k == rounds
    if keep_model:
        shutil.copytree(local / "model", tmp / "model")
    receipt["outputs"] = {p.name: P.sha256_file(p) for p in sorted(tmp.iterdir()) if p.is_file()}
    (tmp / "receipt.json").write_text(json.dumps(receipt, indent=2))
    (tmp / "_SUCCESS").write_text("")
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    shutil.rmtree(local, ignore_errors=True)
    shutil.rmtree(local.parent / f"{local.name}-inputs", ignore_errors=True)
    print(f"[{arm} seed{seed} round{k}] steps {receipt['optimizer_steps']} fit {receipt['fit_seconds']:.0f}s "
          f"score {receipt['score_seconds']:.0f}s{' (model kept)' if keep_model else ''}", flush=True)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, nargs="*", help="default: every round in order")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true", help="2 steps per round, small frames; needs a plan sealed with --smoke")
    a = p.parse_args(argv)
    cfg = json.loads((P.HERE / "config.json").read_text())
    sealed = P.load_plan(a.run_dir)
    if sealed["plan"]["smoke"] != a.smoke:
        raise SystemExit("--smoke must match the plan (smoke plans live in a scratch run folder)")
    if a.seed not in cfg["seeds"] + cfg["extra_seeds_near_boundary"] and not a.smoke:
        raise SystemExit(f"seed {a.seed} is not in config.json")
    device = P.resolve_device(a.device)
    total = sealed["plan"]["config"]["stream"]["rounds"]
    receipts = [run_round(a.repo, cfg, a.data_dir, a.run_dir, sealed, a.arm, a.seed, k, device, a.smoke)
                for k in (a.rounds or range(1, total + 1))]
    print(json.dumps([{k: r[k] for k in ("round", "version", "train_rows", "optimizer_steps", "fit_seconds")}
                      for r in receipts], indent=1))


if __name__ == "__main__":
    main()
