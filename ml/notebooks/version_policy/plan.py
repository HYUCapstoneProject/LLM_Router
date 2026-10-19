"""RF-06 data, pilot and sealed stream plan: version policy (P1) vs full retraining (P2).

build (local, CPU): from the EV-10 00_prepare tables write the experiment tables in the objective-comparison
      training format with sample_weight 1 and a bootstrap/split group id:
        pool.parquet   EV-10 train rows (v2-excluded and two-or-more-missing-tier rows dropped, one missing tier
                       filled from its neighbours as in EV-10 03_ours). Seed, probe windows and reserve come from here.
        dev.parquet    EV-10 dev without imputed rows and evaluation exclusions (monitoring, pilot selection).
        test.parquet   EV-10 test, same rules (final models only; already opened by EV-10 08).
pilot (Colab T4): train one pilot model on a nested, source-stratified, group-complete subset of the pool and
      score dev. The selected size becomes M0.
seal (Colab T4): pick n0 and the stream size from the pilot curve (plan rules), score the probe windows with the
      fixed logger M0, build every round's retraining batch (all wrong probe rows + reserve rows that restore the
      pool label ratio) and write plan/ with SHAs. Nothing is chosen after this.

Pure helpers (nested subsets, batch rule, version rules, training-row composition) are unit-tested in
ml/tests/test_version_policy_experiment.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "router_final"))  # EV-10 label and imputation rules
sys.path.insert(0, str(HERE.parents[1] / "src"))  # ml/src: the objective-comparison trainer and scorer

LABELS = ("easy", "medium", "hard")
TIERS = ("low", "medium", "high")
MAX_LENGTH = 512  # routing_objective_continual_v1.json backbone.max_length
SCORE_BATCH = 32


# ---------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed, values: pd.Series) -> pd.Series:
    """Deterministic pseudo-random order key (independent of row order and Python hash seed)."""
    return values.astype(str).map(lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest())


def label_ratio(frame: pd.DataFrame) -> dict:
    share = frame["difficulty_label"].value_counts(normalize=True)
    return {label: float(share.get(label, 0.0)) for label in LABELS}


def take_groups(frame: pd.DataFrame, quota: dict, seed) -> pd.DataFrame:
    """Per source, whole groups in stable-key order until the source quota of rows is reached.

    Because the order is fixed by `seed`, a smaller quota always gives a subset of a larger one (nested)."""
    parts = []
    for source, rows in frame.groupby("source_dataset", sort=True):
        need = int(quota.get(source, 0))
        if need <= 0:
            continue
        groups = rows.groupby("group_id").size()
        order = stable_key(seed, pd.Series(groups.index, index=groups.index)).sort_values().index
        cum = groups.loc[order].cumsum()
        chosen = cum.index[(cum - groups.loc[order]) < need]
        parts.append(rows[rows.group_id.isin(chosen)])
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0].copy()


def source_quota(frame: pd.DataFrame, n: int) -> dict:
    share = frame["source_dataset"].value_counts(normalize=True)
    return {source: int(round(n * s)) for source, s in share.items()}


def nested_subset(pool: pd.DataFrame, n: int, seed) -> pd.DataFrame:
    return take_groups(pool, source_quota(pool, n), seed)


def validation_rows(dev: pd.DataFrame, per_label: int, seed) -> pd.DataFrame:
    """Fixed in-training validation rows (16 per label in operations), stable-key order."""
    key = stable_key(seed, dev["sample_id"])
    return (dev.assign(_k=key).sort_values("_k").groupby("difficulty_label", sort=True).head(per_label)
            .drop(columns="_k").reset_index(drop=True))


def batch_targets(wrong: dict, ratio: dict) -> dict:
    """Plan rule: N = max_l(e_l / p_l), t_l = max(e_l, round(N * p_l)). Every wrong row is kept."""
    n = max((wrong.get(label, 0) / ratio[label]) for label in LABELS if ratio[label] > 0)
    return {label: int(max(wrong.get(label, 0), round(n * ratio[label]))) for label in LABELS}


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    m = np.zeros((3, 3))
    np.add.at(m, (y_true, y_pred), 1)
    return m


def macro_f1_from_confusion(m: np.ndarray) -> np.ndarray:
    """Macro-F1 over the three labels for one (3, 3) or many (..., 3, 3) confusion matrices (zero division -> 0)."""
    m = np.asarray(m, float)
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    predicted, actual = m.sum(-2), m.sum(-1)
    denom = predicted + actual
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(-1)


def label_ids(labels: pd.Series) -> np.ndarray:
    return labels.map({label: k for k, label in enumerate(LABELS)}).to_numpy(int)


def source_macro_f1(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    """Mean over sources of the per-source three-label macro-F1 (each source counts once)."""
    y = label_ids(frame["difficulty_label"])
    per = {}
    for source in sorted(frame.source_dataset.unique()):
        mask = frame.source_dataset.eq(source).to_numpy()
        per[source] = float(macro_f1_from_confusion(confusion(y[mask], pred[mask])))
    return {"source_macro_f1": float(np.mean(list(per.values()))), "by_source": per}


def select_sizes(curve: dict, majority: float, gap_closure: float, minimum_gain: float) -> dict:
    """Plan 3: n0 = smallest pilot size closing `gap_closure` of the gap between the always-easy baseline and
    the largest pilot size; the stream fills the rest of the measured range (S = n_max - n0) and must add at
    least `minimum_gain` on the pilot curve, otherwise the design is revised before sealing."""
    sizes = sorted(int(n) for n in curve)
    top = sizes[-1]
    target = majority + gap_closure * (curve[top] - majority)
    n0 = next((n for n in sizes if curve[n] >= target - 1e-12), top)  # none: the curve never rose (ok=False)
    gain = curve[top] - curve[n0]
    return {"n0": n0, "n_max": top, "target": target, "stream_rows": top - n0, "pilot_gain": gain,
            "ok": n0 < top and gain >= minimum_gain - 1e-12}


def curve_at(curve: dict, n: float) -> float:
    """Pilot curve interpolated linearly in log(rows)."""
    sizes = sorted(int(k) for k in curve)
    return float(np.interp(np.log(n), np.log(sizes), [curve[k] for k in sizes]))


def window_cap(rest_rows: int, rest_ratio: dict, fill_per_probe_row: dict, rounds: int, safety: float) -> int:
    """Largest window size whose fill demand fits in the reserve left after the probe windows are taken.

    For each label l: rounds*W*fill_l <= (rest - rounds*W) * share_l  =>  W <= rest*share_l / (rounds*(fill_l + share_l)).
    Fill rates are measured on dev, so a safety factor keeps room for the probe's own label mix."""
    caps = [rest_rows * rest_ratio[l] / (rounds * (fill_per_probe_row[l] + rest_ratio[l]))
            for l in LABELS if fill_per_probe_row.get(l, 0) > 0]
    return int(safety * min(caps)) if caps else rest_rows // rounds


def version_name(round_index: int, cycle: int) -> str:
    """Operational successfulHttpRequest numbering: 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, ..."""
    c, j = divmod(round_index - 1, cycle)
    return f"{c + 2}.0" if j + 1 == cycle else f"{c + 1}.{j + 1}"


def anchor_round(round_index: int, cycle: int) -> int:
    """Round whose model is the last approved .0 anchor before this round (0 = M0 = version 1.0)."""
    return ((round_index - 1) // cycle) * cycle


def first_per_label(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Operational compact replay: per final label, stable hash order, at most `cap` rows."""
    if frame.empty:
        return frame
    key = frame["sample_id"].astype(str).map(lambda v: hashlib.sha256(v.encode()).hexdigest())
    return (frame.assign(_k=key).sort_values("_k").groupby("difficulty_label", sort=True).head(cap)
            .drop(columns="_k"))


def p1_rows(round_index: int, batches: pd.DataFrame, seed_rows: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Version policy: current cycle's batches in full + compact history replay + base (D_seed) rehearsal."""
    a = anchor_round(round_index, policy["cycle_size"])
    current = batches[(batches["round"] > a) & (batches["round"] <= round_index)].assign(part="cycle")
    history = first_per_label(batches[batches["round"] <= a], policy["history_replay_per_label"]).assign(part="history")
    base = first_per_label(seed_rows, policy["base_rehearsal_per_label"]).assign(part="base")
    rows = pd.concat([current, history, base], ignore_index=True)
    if rows.sample_id.duplicated().any():
        raise ValueError("P1 training rows repeat a sample")
    return rows


def p2_rows(round_index: int, batches: pd.DataFrame, seed_rows: pd.DataFrame) -> pd.DataFrame:
    """Full retraining: every row seen so far (D_seed and all batches up to this round)."""
    rows = pd.concat([seed_rows.assign(part="seed"),
                      batches[batches["round"] <= round_index].assign(part="batch")], ignore_index=True)
    if rows.sample_id.duplicated().any():
        raise ValueError("P2 training rows repeat a sample")
    return rows


def assign_windows(probe: pd.DataFrame, rounds: int, seed) -> pd.Series:
    """Group-complete, per-source even split of the probe rows into `rounds` arrival windows (1-based)."""
    window = pd.Series(0, index=probe.index)
    for _, rows in probe.groupby("source_dataset", sort=True):
        groups = rows.groupby("group_id").size()
        order = stable_key(seed, pd.Series(groups.index, index=groups.index)).sort_values().index
        before = groups.loc[order].cumsum() - groups.loc[order]
        per = len(rows) / rounds
        index = np.minimum(rounds - 1, (before / per).astype(int)) + 1
        window.loc[rows.index] = rows["group_id"].map(index).to_numpy()
    return window


def build_batches(probe: pd.DataFrame, predicted: np.ndarray, reserve: pd.DataFrame, ratio: dict,
                  rounds: int, seed) -> tuple:
    """For every window: all rows the fixed logger got wrong + reserve rows (stable order, never reused)
    so the batch label mix equals the pool ratio. Raises if the reserve runs out (revise before sealing)."""
    probe = probe.assign(predicted=np.asarray(predicted))
    probe["wrong"] = probe["predicted"].ne(label_ids(probe["difficulty_label"]))
    queue = {label: rows.sample_id.tolist() for label, rows in
             reserve.assign(_k=stable_key(seed, reserve["sample_id"])).sort_values("_k").groupby("difficulty_label")}
    by_id = reserve.set_index("sample_id")
    batches, report = [], []
    for k in range(1, rounds + 1):
        window = probe[probe["window"].eq(k)]
        wrong = window[window["wrong"]]
        counts = wrong["difficulty_label"].value_counts().to_dict()
        targets = batch_targets(counts, ratio)
        fill_ids = []
        for label in LABELS:
            need = targets[label] - counts.get(label, 0)
            if need > len(queue.get(label, [])):
                raise ValueError(f"round {k}: reserve has {len(queue.get(label, []))} {label} rows, needs {need}")
            fill_ids += queue[label][:need]
            queue[label] = queue[label][need:]
        fill = by_id.loc[fill_ids].reset_index()
        batch = pd.concat([wrong.drop(columns=["window", "predicted", "wrong"]).assign(origin="wrong"),
                           fill.assign(origin="fill")], ignore_index=True).assign(round=k)
        batches.append(batch)
        report.append({"round": k, "window_rows": int(len(window)), "wrong": int(len(wrong)),
                       "fill": int(len(fill)), "batch_rows": int(len(batch)),
                       "wrong_by_label": {label: int(counts.get(label, 0)) for label in LABELS},
                       "batch_by_label": {label: int(targets[label]) for label in LABELS},
                       "batch_by_source": batch.source_dataset.value_counts().to_dict()})
    left = {label: len(ids) for label, ids in queue.items()}
    return pd.concat(batches, ignore_index=True), report, left


# ---------------------------------------------------------------------------------------------
# build (local)
# ---------------------------------------------------------------------------------------------

def resolve_inputs(repo: Path, data_root: Path, inputs: dict) -> dict:
    """Code-side files (exclusion lists) come from the checkout, large artifacts may live in another checkout."""
    return {k: (repo / v if (repo / v).exists() else data_root / v) for k, v in inputs.items()}


def group_map(inputs: dict) -> pd.Series:
    parts = []
    for key in ("a_train", "a_dev", "a_test_routerbench", "a_test_xroutebench"):
        parts.append(pd.read_parquet(inputs[key], columns=["sample_id", "duplicate_group_id"])
                     .rename(columns={"duplicate_group_id": "group_id"}))
    for key in ("llmrouterbench", "sprout"):  # EV-10 re-split unit
        parts.append(pd.read_parquet(inputs[key], columns=["sample_id", "split_group_id"])
                     .rename(columns={"split_group_id": "group_id"}))
    g = pd.concat(parts, ignore_index=True).drop_duplicates("sample_id")
    return g.set_index("sample_id")["group_id"].astype(str)


def to_frame(rows: pd.DataFrame, groups: pd.Series, sample_weight: float) -> pd.DataFrame:
    import ours  # EV-10 03_ours objective format
    category = rows["slice"].fillna(rows["source_dataset"]) if "slice" in rows else rows["source_dataset"]
    frame = ours.to_objective_frame(rows.reset_index(drop=True), category.reset_index(drop=True))
    frame["sample_weight"] = float(sample_weight)
    frame["group_id"] = frame["sample_id"].map(groups)
    if frame["group_id"].isna().any():
        raise SystemExit(f"{int(frame.group_id.isna().sum())} rows have no group id")
    return frame


def build(repo: Path, data_root: Path, cfg: dict, output_dir: Path) -> dict:
    import ours
    from prepare import needed_tier
    inputs = resolve_inputs(repo, data_root, cfg["inputs"])
    prep_dir = inputs["prepare_dir"]
    prep = json.loads((prep_dir / "manifest.json").read_text())
    for name in ("train.parquet", "dev.parquet", "test.parquet"):
        if sha256_file(prep_dir / name) != prep["outputs"][name]["sha256"]:
            raise SystemExit(f"00_prepare output changed: {name}")
    tau = float(prep["config"]["tau"])
    groups = group_map(inputs)
    excluded = set()
    for key in ("non_english_exclusions", "document_grounded_exclusions"):
        excluded |= set(pd.read_csv(inputs[key])["sample_id"].astype(str))
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"experiment": cfg["experiment"], "experiment_version": cfg["experiment_version"], "tau": tau,
                "prepare_manifest_sha256": sha256_file(prep_dir / "manifest.json"),
                "config_sha256": sha256_file(HERE / "config.json"), "outputs": {}, "rows": {}, "dropped": {},
                "label_ratio": {}, "rows_by_source": {}}
    # pool: training rows, imputation as EV-10 03_ours
    train = pd.read_parquet(prep_dir / "train.parquet")
    v2 = train["excluded_v2"].fillna(False).astype(bool)
    filled, imputed, sparse = ours.impute_single_missing_tier(train)
    keep = ~v2 & ~sparse
    pool_rows = filled[keep].reset_index(drop=True)
    probs = pool_rows[[f"p_{t}" for t in TIERS]].to_numpy(float)
    pool_rows["needed_tier"] = [TIERS[i] for i in needed_tier(probs, tau)]
    frames = {"pool": to_frame(pool_rows, groups, cfg["sample_weight"])}
    manifest["dropped"]["pool"] = {"v2_excluded": int(v2.sum()), "two_or_more_tiers_missing": int((sparse & ~v2).sum()),
                                   "imputed_one_tier_kept": int((imputed & keep).sum())}
    # evaluation sets: observed tiers only (EV-10 evaluation rule), exclusions dropped
    for role in ("dev", "test"):
        rows = pd.read_parquet(prep_dir / f"{role}.parquet")
        v2 = rows["excluded_v2"].fillna(False).astype(bool)
        missing = rows[[f"p_{t}" for t in TIERS]].isna().any(axis=1)
        out = rows["sample_id"].astype(str).isin(excluded)
        manifest["dropped"][role] = {"v2_excluded": int(v2.sum()), "missing_tier": int((missing & ~v2).sum()),
                                     "evaluation_exclusions": int((out & ~v2 & ~missing).sum())}
        frames[role] = to_frame(rows[~v2 & ~missing & ~out], groups, cfg["sample_weight"])
    for name, frame in frames.items():
        path = output_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        manifest["outputs"][path.name] = sha256_file(path)
        manifest["rows"][name] = int(len(frame))
        manifest["rows_by_source"][name] = frame.source_dataset.value_counts().sort_index().to_dict()
        manifest["label_ratio"][name] = {k: round(v, 4) for k, v in label_ratio(frame).items()}
    (output_dir / "outcome-manifest.json").write_text(json.dumps(
        {"dataset_version": cfg["experiment_version"], "prepare_manifest_sha256": manifest["prepare_manifest_sha256"]},
        indent=2))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def load_data(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        if sha256_file(data_dir / name) != digest:
            raise SystemExit(f"data changed: {name}")
    return {"manifest": manifest, **{name: pd.read_parquet(data_dir / f"{name}.parquet") for name in ("pool", "dev", "test")}}


# ---------------------------------------------------------------------------------------------
# Colab: training and scoring (shared with rounds.py)
# ---------------------------------------------------------------------------------------------

def resolve_device(device: str) -> str:
    import torch
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


def score(model_dir: Path, frame: pd.DataFrame, device: str) -> pd.DataFrame:
    """Raw logits (low/medium/high) with the training tokenisation (512, right truncation), length-batched."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from llm_router_labeling.objective_comparison.training import _predict_logits
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    text = frame["model_input_text"].astype(str)
    order = np.argsort(text.str.len().to_numpy(), kind="stable")
    logits = np.zeros((len(frame), 3))
    logits[order] = _predict_logits(model, tokenizer, frame.iloc[order][["model_input_text"]], batch_size=SCORE_BATCH,
                                    max_length=MAX_LENGTH, device=device)
    out = frame[["sample_id"]].copy()
    for k, tier in enumerate(TIERS):
        out[f"score_{tier}"] = logits[:, k]
    return out


def token_count(model_dir_or_id: str, texts: pd.Series, revision: str | None = None) -> int:
    """Tokens the trainer sees per epoch (truncated at 512)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id, revision=revision)
    ids = tokenizer(texts.astype(str).tolist(), truncation=True, max_length=MAX_LENGTH)["input_ids"]
    return int(sum(len(x) for x in ids))


def train_model(repo: Path, cfg: dict, rows: pd.DataFrame, validation: pd.DataFrame, outcome_manifest: Path,
                output_dir: Path, seed: int, device: str, initial_model_dir: Path | None = None,
                maximum_train_steps: int = 0, smoke: bool = False) -> dict:
    """One training run with the EV-10 / RT-05 trainer and settings; weights from `initial_model_dir` if given."""
    from llm_router_labeling.objective_comparison.training import train_objective
    if output_dir.exists():
        shutil.rmtree(output_dir)
    inputs = output_dir.parent / f"{output_dir.name}-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    columns = [c for c in validation.columns if c not in ("group_id",)]
    rows[columns].to_parquet(inputs / "train.parquet", index=False)
    validation[columns].to_parquet(inputs / "validation.parquet", index=False)
    objective = json.loads((repo / cfg["objective_config"]).read_text())
    backbone = objective["backbone"]
    tokens = token_count(str(initial_model_dir) if initial_model_dir else backbone["model_id"], rows["model_input_text"],
                         None if initial_model_dir else backbone["revision"])
    result = train_objective(
        method=cfg["method"], training_path=inputs / "train.parquet", validation_path=inputs / "validation.parquet",
        outcome_manifest_path=outcome_manifest, experiment_config_path=repo / cfg["objective_config"],
        output_dir=output_dir, seed=seed, device=device, initial_model_dir=initial_model_dir,
        maximum_train_rows=64 if smoke else 0, maximum_validation_rows=48 if smoke else 0,
        maximum_train_steps=2 if smoke else maximum_train_steps)
    metrics = result["metrics"]
    epochs = int(objective["training"]["epochs"])
    return {"train_rows": int(metrics["train_rows"]), "optimizer_steps": int(metrics["optimizer_steps"]),
            "fit_seconds": float(metrics["fit_seconds"]), "epochs": epochs,
            "tokens_per_epoch": tokens, "planned_samples": int(len(rows)) * epochs,
            "training_data_sha256": result["manifest"]["training_data_sha256"],
            "initial_model_sha256": result["manifest"]["initial_model_sha256"],
            "model_sha256": sha256_file(output_dir / "model" / "model.safetensors"),
            "peak_vram_mb": metrics["peak_vram_mb"], "device": metrics["device"]}


# ---------------------------------------------------------------------------------------------
# pilot and seal (Colab)
# ---------------------------------------------------------------------------------------------

def smoke_view(cfg: dict) -> dict:
    """Tiny sizes for the smoke runs (2 training steps, small frames); never written to the run folder."""
    out = json.loads(json.dumps(cfg))
    out["pilot"]["sizes"] = [200, 400]
    out["stream"]["rounds"] = 4  # one round past the first anchor (2.0 -> 2.1)
    return out


def pilot(repo: Path, cfg: dict, data_dir: Path, run_dir: Path, size: int, device: str, smoke: bool) -> dict:
    target = run_dir / "pilot" / f"n{size}"
    if (target / "_SUCCESS").exists():
        print(f"[pilot n{size}] done", flush=True)
        return json.loads((target / "receipt.json").read_text())
    data = load_data(data_dir)
    rows = nested_subset(data["pool"], size, cfg["plan_seed"])
    validation = validation_rows(data["dev"], cfg["validation_rows_per_label"], cfg["plan_seed"])
    local = Path("/content/rf06-work" if Path("/content").exists() else run_dir / "_work") / f"pilot-n{size}"
    receipt = train_model(repo, cfg, rows, validation, data_dir / "outcome-manifest.json", local,
                          cfg["pilot"]["seed"], device, smoke=smoke)
    dev = data["dev"] if not smoke else data["dev"].groupby("source_dataset").head(40)
    scores = score(local / "model", dev, device)
    pred = scores[[f"score_{t}" for t in TIERS]].to_numpy().argmax(1)
    receipt.update({"size": size, "actual_rows": int(len(rows)), "rows_by_source": rows.source_dataset.value_counts().to_dict(),
                    "label_mix": label_ratio(rows), "dev": source_macro_f1(dev.reset_index(drop=True), pred)})
    target.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(target / "dev-scores.parquet", index=False)
    rows[["sample_id"]].to_parquet(target / "rows.parquet", index=False)
    shutil.copytree(local / "model", target / "model")
    (target / "receipt.json").write_text(json.dumps(receipt, indent=2))
    (target / "_SUCCESS").write_text("")
    shutil.rmtree(local, ignore_errors=True)
    print(f"[pilot n{size}] rows {len(rows)} steps {receipt['optimizer_steps']} fit {receipt['fit_seconds']:.0f}s "
          f"dev source-macro-F1 {receipt['dev']['source_macro_f1']:.4f}", flush=True)
    return receipt


def majority_baseline(dev: pd.DataFrame) -> float:
    return source_macro_f1(dev, np.zeros(len(dev), int))["source_macro_f1"]


def seal(repo: Path, cfg: dict, data_dir: Path, run_dir: Path, device: str, smoke: bool,
         n0_override: int | None = None) -> dict:
    plan_dir = run_dir / "plan"
    if (plan_dir / "_SUCCESS").exists():
        raise SystemExit("plan already sealed")
    data = load_data(data_dir)
    pool, dev = data["pool"], data["dev"]
    if smoke:
        dev = dev.groupby("source_dataset").head(40).reset_index(drop=True)
    sizes = cfg["pilot"]["sizes"]
    receipts = {n: json.loads((run_dir / "pilot" / f"n{n}" / "receipt.json").read_text()) for n in sizes}
    curve = {n: r["dev"]["source_macro_f1"] for n, r in receipts.items()}
    majority = majority_baseline(dev)
    chosen = select_sizes(curve, majority, cfg["pilot"]["gap_closure"], cfg["pilot"]["minimum_stream_gain"])
    if n0_override is not None:  # smoke only
        chosen.update({"n0": n0_override, "stream_rows": chosen["n_max"] - n0_override, "ok": True})
    print(json.dumps({"curve": curve, "always_easy": majority, **chosen}, indent=1), flush=True)
    if not chosen["ok"]:
        raise SystemExit("pilot rule failed: the stream would add less than the minimum gain (or n0 = n_max). "
                         "Stop and revise the design before sealing.")
    n0, rounds = chosen["n0"], cfg["stream"]["rounds"]
    m0_dir = run_dir / "pilot" / f"n{n0}" / "model"
    ratio = label_ratio(pool)
    # batch rows per probe row, measured by applying the batch rule to M0's dev predictions
    m0_dev = pd.read_parquet(run_dir / "pilot" / f"n{n0}" / "dev-scores.parquet").set_index("sample_id").loc[dev.sample_id]
    dev_pred = m0_dev[[f"score_{t}" for t in TIERS]].to_numpy().argmax(1)
    dev_wrong = dev[dev_pred != label_ids(dev["difficulty_label"])]["difficulty_label"].value_counts().to_dict()
    dev_targets = batch_targets(dev_wrong, ratio)
    per_probe_row = sum(dev_targets.values()) / len(dev)
    fill_per_probe_row = {l: (dev_targets[l] - dev_wrong.get(l, 0)) / len(dev) for l in LABELS}
    window_rows = int(math.ceil(chosen["stream_rows"] / (rounds * per_probe_row)))
    # partition: D_seed = the pilot subset, probe windows next, the rest is the reserve
    seed_rows = nested_subset(pool, n0, cfg["plan_seed"])
    rest = pool[~pool.group_id.isin(set(seed_rows.group_id))]
    cap = window_cap(len(rest), label_ratio(rest), fill_per_probe_row, rounds, cfg["stream"]["reserve_safety"])
    capped = window_rows > cap
    if capped:  # the pool cannot supply the planned stream: shorten it and re-check the gain rule
        window_rows = cap
        stream = rounds * window_rows * per_probe_row
        gain = curve_at(curve, n0 + stream) - curve[n0]
        chosen.update({"stream_rows_capped": stream, "pilot_gain_capped": gain})
        print(f"[seal] window capped by the reserve: {cap} rows, stream ~{stream:.0f}, interpolated gain {gain:.4f}", flush=True)
        if gain < cfg["pilot"]["minimum_stream_gain"] - 1e-12 and not smoke:
            raise SystemExit("the pool cannot supply a stream that adds the minimum gain. Stop and revise the design.")
    probe = take_groups(rest, source_quota(rest, rounds * window_rows), f"{cfg['plan_seed']}:probe")
    probe = probe.assign(window=assign_windows(probe, rounds, f"{cfg['plan_seed']}:window").to_numpy())
    reserve = rest[~rest.group_id.isin(set(probe.group_id))]
    print(f"[seal] n0 {len(seed_rows)}, window {window_rows} x {rounds}, probe {len(probe)}, reserve {len(reserve)}", flush=True)
    probe_scores = score(m0_dir, probe, device)
    probe_pred = probe_scores[[f"score_{t}" for t in TIERS]].to_numpy().argmax(1)
    batches, report, reserve_left = build_batches(probe.reset_index(drop=True), probe_pred, reserve, ratio, rounds,
                                                  f"{cfg['plan_seed']}:fill")
    validation = validation_rows(data["dev"], cfg["validation_rows_per_label"], cfg["plan_seed"])
    # P0 = M0 on test (texts only; outcomes are read in 03_compare)
    test = data["test"] if not smoke else data["test"].groupby("source_dataset").head(40)
    m0_test = score(m0_dir, test, device)
    tmp = plan_dir.parent / "plan-tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files = {"seed.parquet": seed_rows, "probe.parquet": probe, "batches.parquet": batches,
             "validation.parquet": validation, "m0-probe-scores.parquet": probe_scores,
             "m0-test-scores.parquet": m0_test}
    for name, frame in files.items():
        frame.to_parquet(tmp / name, index=False)
    shutil.copyfile(run_dir / "pilot" / f"n{n0}" / "dev-scores.parquet", tmp / "m0-dev-scores.parquet")
    plan = {"experiment": cfg["experiment"], "experiment_version": cfg["experiment_version"], "smoke": smoke,
            "data_manifest_sha256": sha256_file(data_dir / "manifest.json"), "config": cfg,
            "pilot_curve": {str(n): curve[n] for n in sizes}, "always_easy": majority, "selection": chosen,
            "pilot_receipts": {str(n): {k: receipts[n][k] for k in ("actual_rows", "optimizer_steps", "fit_seconds")}
                               for n in sizes},
            "label_ratio": ratio, "batch_rows_per_probe_row_dev": per_probe_row,
            "fill_rows_per_probe_row_dev": fill_per_probe_row, "window_rows": window_rows, "window_capped": capped,
            "rows": {"seed": int(len(seed_rows)), "probe": int(len(probe)), "reserve": int(len(reserve)),
                     "reserve_left": reserve_left, "batches": int(len(batches))},
            "rounds": report, "m0_model_sha256": sha256_file(m0_dir / "model.safetensors"),
            "versions": {k: version_name(k, cfg["version_policy"]["cycle_size"]) for k in range(1, rounds + 1)},
            "files": {name: sha256_file(tmp / name) for name in sorted(p.name for p in tmp.iterdir())}}
    (tmp / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    shutil.copytree(m0_dir, tmp / "m0-model")
    (tmp / "_SUCCESS").write_text(sha256_file(tmp / "plan.json") + "\n")
    tmp.rename(plan_dir)
    return plan


def load_plan(run_dir: Path) -> dict:
    plan_dir = run_dir / "plan"
    if not (plan_dir / "_SUCCESS").exists():
        raise SystemExit("plan is not sealed (run 00 first)")
    if sha256_file(plan_dir / "plan.json") != (plan_dir / "_SUCCESS").read_text().split()[0]:
        raise SystemExit("plan.json changed after sealing")
    plan = json.loads((plan_dir / "plan.json").read_text())
    for name, digest in plan["files"].items():
        if sha256_file(plan_dir / name) != digest:
            raise SystemExit(f"plan file changed: {name}")
    if sha256_file(plan_dir / "m0-model" / "model.safetensors") != plan["m0_model_sha256"]:
        raise SystemExit("M0 model changed")
    frames = {name[:-8]: pd.read_parquet(plan_dir / name) for name in plan["files"] if name.endswith(".parquet")}
    return {"plan": plan, "dir": plan_dir, **frames}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--repo", type=Path, required=True)
    b.add_argument("--data-root", type=Path, default=None, help="checkout holding ml/artifacts (default: --repo)")
    b.add_argument("--output-dir", type=Path, required=True)
    for name in ("pilot", "seal"):
        s = sub.add_parser(name)
        s.add_argument("--repo", type=Path, required=True)
        s.add_argument("--data-dir", type=Path, required=True)
        s.add_argument("--run-dir", type=Path, required=True)
        s.add_argument("--device", default="auto")
        s.add_argument("--smoke", action="store_true", help="tiny sizes, 2 steps; use a scratch --run-dir")
        if name == "pilot":
            s.add_argument("--sizes", type=int, nargs="*", help="default: every pilot size in config.json")
        else:
            s.add_argument("--n0", type=int, default=None, help="smoke only: force n0")
    a = p.parse_args(argv)
    cfg = json.loads((HERE / "config.json").read_text())
    if getattr(a, "smoke", False):
        cfg = smoke_view(cfg)
    if a.command == "build":
        out = build(a.repo, a.data_root or a.repo, cfg, a.output_dir)
    elif a.command == "pilot":
        device = resolve_device(a.device)
        start = time.time()
        out = {n: pilot(a.repo, cfg, a.data_dir, a.run_dir, n, device, a.smoke)["dev"]["source_macro_f1"]
               for n in (a.sizes or cfg["pilot"]["sizes"])}
        out["seconds"] = time.time() - start
    else:
        if a.n0 is not None and not a.smoke:
            raise SystemExit("--n0 is for smoke runs only; the plan rule picks n0")
        plan = seal(a.repo, cfg, a.data_dir, a.run_dir, resolve_device(a.device), a.smoke, a.n0)
        out = {k: plan[k] for k in ("selection", "window_rows", "rows", "versions")}
        out["rounds"] = [{k: r[k] for k in ("round", "wrong", "fill", "batch_rows")} for r in plan["rounds"]]
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
