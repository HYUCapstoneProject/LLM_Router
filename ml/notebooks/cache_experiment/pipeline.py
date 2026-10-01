"""Stage functions called by the notebooks. Each stage seals what it writes.

Layout under <data_root>/ml/artifacts/cache-experiment-v1/<run_id>/:
  prepare/  pool, exclusions, embeddings, token lengths, scorer self-check, parity
  d1/       traces/<trace>/trace.parquet and <trace>/<arm>/requests.parquet, summary
  d2/       originals, human sheets, frozen annotations, calibration, frozen policy, test
  latency/  model and pgvector timings, HNSW recall, pgvector decision parity
  report/   tables, figures, REPORT-draft.md, results-for-review.zip
"""
from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cache_experiment as ce

STAGES = ("prepare", "d1", "d2", "latency", "report")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    def __init__(self, data_root, run_id, config_path=ce.HERE / "experiment_config.json", output_root=None):
        self.data_root = Path(data_root).resolve()
        self.config_path = Path(config_path)
        self.config = ce.load_config(self.config_path)
        self.run_id = run_id
        base = Path(output_root) if output_root else self.data_root / "ml/artifacts" / self.config["experiment_id"]
        self.root = base.resolve() / run_id
        for stage in STAGES:
            (self.root / stage).mkdir(parents=True, exist_ok=True)

    def path(self, *parts):
        return self.root.joinpath(*parts)

    def config_pin(self):
        return {"path": str(self.config_path.name), "sha256": ce.sha(self.config_path)}

    def require(self, *parts):
        path = self.path(*parts)
        if not path.exists():
            raise FileNotFoundError(f"Run the earlier stage first: {path}")
        return path

    def pool(self):
        return ce.read_parquet(self.require("prepare", "pool.parquet"))

    def embeddings(self):
        vectors = np.load(self.require("prepare", "embeddings.npy"))
        manifest = ce.read(self.require("prepare", "prepare-manifest.json"))
        if ce.sha(self.path("prepare", "embeddings.npy")) != manifest["artifacts"]["embeddings.npy"]:
            raise ValueError("embeddings.npy differs from the sealed prepare manifest")
        return vectors

    def d2_originals(self):
        return ce.read_parquet(self.require("d2", "originals.parquet"))

    def d1_pool(self):
        pool = self.pool()
        taken = set(self.d2_originals().sample_id)
        return pool[~pool.sample_id.isin(taken)].reset_index(drop=True), pool


# ------------------------------------------------------------------ prepare --

def copy_logger(run, drive_logger_path):
    """Copy the retraining-effect logger log into the data root and pin it."""
    target = run.data_root / run.config["inputs"]["logger_log"]["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if not drive_logger_path or not Path(drive_logger_path).is_file():
            raise FileNotFoundError("logger-log.parquet not found; set CACHE_EXP_DRIVE_LOGGER to its path")
        shutil.copy2(drive_logger_path, target)
    ce.file_pin(target, run.config["inputs"]["logger_log"]["sha256"])
    return target


def load_inputs(run):
    cfg, root = run.config, run.data_root
    paths = {name: ce.input_path(cfg, name, root) for name in cfg["inputs"]}
    stream = pd.read_parquet(paths["stream"])
    logger = pd.read_parquet(paths["logger_log"])
    routerbench = pd.read_pickle(paths["routerbench_5shot"])
    xroute = pd.concat([pd.read_parquet(paths["xroutebench_train"]), pd.read_parquet(paths["xroutebench_test"])],
                       ignore_index=True)
    return paths, stream, logger, routerbench, xroute


def scorer_selfcheck(pool):
    """Score each routed answer against its own question as if it were reused.

    Agreement with the recorded success is the evidence that the reuse scorer
    reads answers the way the benchmark did. Disagreements are reported, not fixed.
    """
    rows = []
    for r in pool[pool.task_family.isin(["mcq", "numeric"])].itertuples():
        entry = {"duplicate_group_id": "_entry", "key_hash": "_entry", "routed_answer": r.routed_answer,
                 "options_json": r.options_json}
        target = {"duplicate_group_id": "_target", "key_hash": "_target", "task_family": r.task_family, "gold": r.gold,
                  "options_json": r.options_json, "source_dataset": r.source_dataset}
        rows.append({"sample_id": r.sample_id, "source_dataset": r.source_dataset, "task_family": r.task_family,
                     "scored": ce.reuse_outcome(entry, target), "recorded_success": float(r.routed_success)})
    frame = pd.DataFrame(rows)
    table = frame.groupby(["source_dataset", "task_family", "scored", "recorded_success"]).size().rename("rows").reset_index()
    decided = frame[frame.scored.isin([ce.REUSE_CORRECT, ce.REUSE_INCORRECT])]
    agree = (decided.scored.eq(ce.REUSE_CORRECT) == decided.recorded_success.eq(1.0))
    return {"rows": int(len(frame)), "decided": int(len(decided)), "unknown": int(len(frame) - len(decided)),
            "agreement_on_decided": float(agree.mean()) if len(decided) else float("nan"),
            "disagreements": int((~agree).sum()), "table": table.to_dict("records")}, frame


def route_requests(run, stream, logger, router, progress=500):
    """Service-shaped routing plus the evidence that the local router is the anchor.

    The logger classified `model_input_text` (the training input). The service
    classifies the raw message, which here is the request text used as the cache
    key. First the local router must reproduce the logger on the logger's own
    input for a fixed sample (gate); then every request is routed on its text.
    """
    cfg = run.config["router"]
    sample = stream.merge(logger[["sample_id", "initial_label"]], on="sample_id").sample(
        n=cfg["parity_sample"], random_state=run.config["bootstrap"]["seed"])
    local = [router.classify(t) for t in sample.model_input_text]
    agreement = float((pd.Series(local, index=sample.index) == sample.initial_label).mean())
    gate = {"rows": int(len(sample)), "agreement": agreement, "minimum": cfg["parity_min_agreement"],
            "passed": agreement >= cfg["parity_min_agreement"]}
    if not gate["passed"]:
        raise RuntimeError(f"Local router does not reproduce the anchor logger: {gate}")
    field = cfg["request_input"]
    labels = []
    for i, text in enumerate(stream[field]):
        labels.append(router.classify(text))
        if progress and (i + 1) % progress == 0:
            print(f"routed {i + 1}/{len(stream)}")
    routed = pd.DataFrame({"sample_id": stream.sample_id, "initial_label": labels})
    joined = routed.merge(logger[["sample_id", "initial_label"]], on="sample_id", suffixes=("", "_logger"))
    diagnostic = {"input": field, "rows": int(len(joined)),
                  "agreement_with_logger_training_input": float((joined.initial_label == joined.initial_label_logger).mean()),
                  "by_label": pd.crosstab(joined.initial_label_logger, joined.initial_label).to_dict()}
    return routed, gate, diagnostic


def prepare(run, embedder, reranker, router, runtime, embed_progress=500):
    cfg = run.config
    paths, stream, logger, routerbench, xroute = load_inputs(run)
    routed, router_gate, router_diagnostic = route_requests(run, stream, logger, router)
    pool, exclusions = ce.build_pool(stream, routed, routerbench, xroute, cfg)
    # The service rejects a message over the BGE input limit instead of truncating
    # it, so such a question is never cached or answered: it is not cache traffic.
    tokens = pd.DataFrame({"sample_id": pool.sample_id, "tokens": [embedder.token_count(t) for t in pool.request_text]})
    over = tokens.tokens.to_numpy() > cfg["service_input_tokens"]
    exclusions = pd.concat([exclusions, pd.DataFrame({"sample_id": pool.sample_id[over],
                                                      "reason": f"exceeds_service_input_limit_{cfg['service_input_tokens']}"})],
                           ignore_index=True)
    pool = pool[~over].reset_index(drop=True)
    tokens = tokens[~over].reset_index(drop=True)
    selfcheck, selfcheck_rows = scorer_selfcheck(pool)

    originals = ce.select_d2_originals(pool, cfg, dict(zip(tokens.sample_id, tokens.tokens)))

    questions = json.loads(paths["production_questions"].read_text())
    probes = json.loads(paths["production_probes"].read_text())
    import models
    parity, parity_rows = models.check_production_parity(questions, probes, embedder, reranker, cfg)
    if not parity["passed"]:
        raise RuntimeError(f"Local models do not reproduce the deployed stack: {parity}")

    vectors = embedder.embed_all(pool.request_text.tolist(), progress=embed_progress)

    out = {}
    out["pool.parquet"] = ce.write_parquet_once(run.path("prepare", "pool.parquet"), pool)
    out["exclusions.parquet"] = ce.write_parquet_once(run.path("prepare", "exclusions.parquet"), exclusions)
    out["tokens.parquet"] = ce.write_parquet_once(run.path("prepare", "tokens.parquet"), tokens)
    out["routing.parquet"] = ce.write_parquet_once(run.path("prepare", "routing.parquet"), routed)
    out["scorer-selfcheck.parquet"] = ce.write_parquet_once(run.path("prepare", "scorer-selfcheck.parquet"), selfcheck_rows)
    out["parity.parquet"] = ce.write_parquet_once(run.path("prepare", "parity.parquet"), parity_rows)
    emb = run.path("prepare", "embeddings.npy")
    if emb.exists():
        if not np.array_equal(np.load(emb), vectors):
            raise ValueError("Existing embeddings differ; use a new run directory")
    else:
        np.save(emb, vectors)
    out["embeddings.npy"] = ce.sha(emb)
    out["d2/originals.parquet"] = ce.write_parquet_once(run.path("d2", "originals.parquet"), originals)
    sheet = ce.authoring_sheet(originals)
    sheet_path = run.path("d2", "authoring-sheet.csv")
    if not sheet_path.exists():
        sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")
    out["d2/authoring-sheet.csv"] = ce.sha(sheet_path)

    manifest = {
        "stage": "prepare", "created_at": now(), "config": run.config_pin(), "runtime": runtime,
        "devices": {"embed": str(embedder.device), "rerank": str(reranker.device), "router": str(router.device),
                    "router_package": router.path},
        "inputs": {k: {"path": str(Path(v).relative_to(run.data_root)), "sha256": ce.sha(v)} for k, v in paths.items()},
        "pool": ce.pool_summary(pool), "exclusions": exclusions.reason.value_counts().to_dict(),
        "scorer_selfcheck": {k: v for k, v in selfcheck.items() if k != "table"},
        "production_parity": parity,
        "router_anchor_parity": router_gate, "router_input_diagnostic": router_diagnostic,
        "embedding_norm_range": [float(np.linalg.norm(vectors, axis=1).min()), float(np.linalg.norm(vectors, axis=1).max())],
        "d2_originals": {"groups": int(len(originals)), "by_stratum": originals.stratum.value_counts().sort_index().to_dict(),
                         "by_split": originals.d2_split.value_counts().sort_index().to_dict()},
        "d1_pool_rows": int(len(pool) - len(originals)),
        "artifacts": out,
    }
    ce.exact_json(run.path("prepare", "prepare-manifest.json"), manifest)
    ce.exact_json(run.path("prepare", "scorer-selfcheck.json"), selfcheck)
    return manifest


# ----------------------------------------------------------------------- D1 --

def make_traces(run):
    d1_pool, _ = run.d1_pool()
    cfg = run.config["d1"]
    rows = []
    for seed in cfg["trace_seeds"]:
        for r in cfg["repeat_fractions"]:
            trace = ce.make_trace(d1_pool.sample_id.to_numpy(), cfg["trace_length"], r, seed)
            tid = ce.trace_id(r, seed)
            digest = ce.write_parquet_once(run.path("d1", "traces", tid, "trace.parquet"), trace)
            meta = d1_pool.set_index("sample_id").loc[trace.sample_id]
            manifest = {"trace_id": tid, "repeat_fraction": r, "seed": seed, "length": cfg["trace_length"],
                        "repeats": int(trace.is_repeat.sum()), "unique_questions": int(trace.sample_id.nunique()),
                        "trace_sha256": digest,
                        "requests_by_source": meta.source_dataset.value_counts().sort_index().to_dict(),
                        "requests_by_routed_tier": meta.routed_tier.value_counts().sort_index().to_dict(),
                        "requests_by_family": meta.task_family.value_counts().sort_index().to_dict()}
            ce.exact_json(run.path("d1", "traces", tid, "trace-manifest.json"), manifest)
            rows.append(manifest)
    return pd.DataFrame(rows)


def run_d1(run, reranker, arms=ce.ARMS, policies=None, tag="production", capacities=True):
    """Replay every trace through each arm; C1 is checked against the stop rule."""
    d1_pool, _ = run.d1_pool()
    all_vectors = run.embeddings()
    pool_full = run.pool()
    index = {s: i for i, s in enumerate(pool_full.sample_id)}
    vectors = all_vectors[[index[s] for s in d1_pool.sample_id]]
    policies = policies or ce.config_policies(run.config)
    scorer = ce.PairScorer(reranker.score, d1_pool.request_text.tolist()) if reranker is not None else None
    cfg = run.config["d1"]
    summaries = []
    for seed in cfg["trace_seeds"]:
        for r in cfg["repeat_fractions"]:
            tid = ce.trace_id(r, seed)
            trace = ce.read_parquet(run.require("d1", "traces", tid, "trace.parquet"))
            variants = [(name, policies[name]) for name in arms]
            if capacities:
                variants += [(f"{name}-cap{c}", ce.Policy(**{**policies[name].__dict__, "name": f"{name}-cap{c}", "capacity": c}))
                             for name in cfg["capacity_arms"] if name in policies for c in cfg["capacities"]]
            logs = {}
            for name, policy in variants:
                out = run.path("d1", "traces", tid, name, "requests.parquet")
                if out.exists():
                    log = ce.read_parquet(out)
                else:
                    log = ce.replay(trace, d1_pool, policy, vectors, scorer)
                    evictions = log.attrs.get("evictions", 0)
                    ce.write_parquet_once(out, log)
                    ce.exact_json(run.path("d1", "traces", tid, name, "arm.json"),
                                  {"arm": name, "policy": {k: v for k, v in policy.__dict__.items()},
                                   "evictions": int(evictions), "tag": tag})
                ce.exact_arm_guard(log)
                logs[name] = log
            baseline = ce.group_sums(logs["C0"]).sum() if "C0" in logs else ce.group_sums(
                ce.read_parquet(run.require("d1", "traces", tid, "C0", "requests.parquet"))).sum()
            for name, log in logs.items():
                arm_meta = ce.read(run.path("d1", "traces", tid, name, "arm.json"))
                summaries.append({"trace_id": tid, "repeat_fraction": r, "seed": seed, "arm": name, "tag": tag,
                                  "capacity": arm_meta["policy"].get("capacity"), "evictions": arm_meta["evictions"],
                                  **ce.ratios(ce.group_sums(log).sum(), baseline)})
    summary = pd.DataFrame(summaries)
    ce.write_parquet_once(run.path("d1", f"summary-{tag}.parquet"), summary)
    return summary


def d1_bootstrap(run, arms=ce.ARMS, tag="production"):
    cfg, boot = run.config["d1"], run.config["bootstrap"]
    frames = []
    for seed in cfg["trace_seeds"]:
        for r in cfg["repeat_fractions"]:
            tid = ce.trace_id(r, seed)
            logs = {a: ce.read_parquet(run.require("d1", "traces", tid, a, "requests.parquet")) for a in arms}
            b = ce.paired_bootstrap(logs, "C0", boot["draws"], [boot["seed"], seed, int(r * 100)])
            frames.append(b.assign(trace_id=tid, repeat_fraction=r, seed=seed))
    out = pd.concat(frames, ignore_index=True)
    ce.write_parquet_once(run.path("d1", f"bootstrap-{tag}.parquet"), out)
    return out


# ----------------------------------------------------------------------- D2 --

def d2_read_authoring(run, filled_csv):
    originals = run.d2_originals()
    sheet = pd.read_csv(filled_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = set(ce.AUTHORING_COLUMNS) - set(sheet.columns)
    if missing:
        raise ValueError(f"Authoring sheet lost columns {missing}")
    if set(sheet.group_id) != set(originals.group_id):
        raise ValueError("Authoring sheet groups differ from the sealed originals")
    return originals, sheet


def d2_validate_authoring(run, filled_csv, embedder):
    originals, sheet = d2_read_authoring(run, filled_csv)
    problems = ce.validate_authoring(sheet, originals, token_count=embedder.token_count,
                                     limit=run.config["service_input_tokens"])
    return problems


def d2_make_annotation_sheets(run, filled_csv, embedder):
    problems = d2_validate_authoring(run, filled_csv, embedder)
    if len(problems):
        raise ValueError(f"Fix the authoring sheet first:\n{problems.to_string()}")
    originals, sheet = d2_read_authoring(run, filled_csv)
    variants = ce.variants_from(sheet, originals)
    ce.write_parquet_once(run.path("d2", "variants.parquet"), variants)
    if not run.path("d2", "authoring-filled.csv").exists():
        shutil.copyfile(filled_csv, run.path("d2", "authoring-filled.csv"))
    elif ce.sha(filled_csv) != ce.sha(run.path("d2", "authoring-filled.csv")):
        raise ValueError("Variants were already sealed from a different authoring sheet; use a new run directory")
    blank = ce.annotation_sheet(variants, originals, run.config["d2"]["annotation_seed"])
    for who in ("A", "B"):
        path = run.path("d2", f"annotation-{who}.csv")
        if not path.exists():
            blank.to_csv(path, index=False, encoding="utf-8-sig")
    ce.exact_json(run.path("d2", "variants-manifest.json"), {
        "created_at": now(), "authoring_sha256": ce.sha(run.path("d2", "authoring-filled.csv")),
        "variants_sha256": ce.sha(run.path("d2", "variants.parquet")), "items": int(len(variants)),
        "by_role": variants.role.value_counts().to_dict(), "authors": sorted(set(variants.author))})
    return variants


def _read_annotation(path):
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False).assign(
        semantic_equivalent=lambda f: f.semantic_equivalent.str.strip().str.upper(),
        cached_answer_correct_for_target=lambda f: f.cached_answer_correct_for_target.str.strip().str.upper())


def d2_merge_annotations(run, filled_a, filled_b, adjudication_csv=None):
    """Agreement report; returns items still needing adjudication (empty = done)."""
    variants = ce.read_parquet(run.require("d2", "variants.parquet"))
    seed = run.config["d2"]["annotation_seed"]
    a = ce.unblind(_read_annotation(filled_a), variants, seed)
    b = ce.unblind(_read_annotation(filled_b), variants, seed)
    if set(a.item_id) != set(variants.item_id) or set(b.item_id) != set(variants.item_id):
        raise ValueError("Annotation sheet items differ from the sealed variants")
    adjudication = None
    if adjudication_csv is not None:
        adjudication = pd.read_csv(adjudication_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        adjudication = ce.unblind(adjudication, variants, seed)
        adjudication["final"] = adjudication.final.str.strip().str.upper()
    merged, pending, agreement = ce.merge_annotations(a, b, adjudication)
    if len(pending):
        sheet = pending.merge(a[["item_id", "original_text", "variant_text", "cached_answer", "original_gold"]], on="item_id")
        labels_a = a.set_index("item_id")
        labels_b = b.set_index("item_id")
        sheet["label_a"] = [labels_a.loc[i, f] for i, f in zip(sheet.item_id, sheet.field)]
        sheet["label_b"] = [labels_b.loc[i, f] for i, f in zip(sheet.item_id, sheet.field)]
        sheet["rationale_a"] = [labels_a.loc[i, "rationale"] for i in sheet.item_id]
        sheet["rationale_b"] = [labels_b.loc[i, "rationale"] for i in sheet.item_id]
        sheet["final"], sheet["reason"], sheet["adjudicator"] = "", "", ""
        sheet["item_id"] = [ce.blind_id(i, seed) for i in sheet.item_id]
        sheet.to_csv(run.path("d2", "adjudication-needed.csv"), index=False, encoding="utf-8-sig")
        return agreement, pending
    if merged[["semantic_equivalent", "cached_answer_correct_for_target"]].isna().any().any():
        raise ValueError("Some final labels are still empty")
    final = variants.merge(merged, on="item_id", validate="one_to_one")
    digest = ce.write_parquet_once(run.path("d2", "annotations.parquet"), final)
    for name, path in (("annotation-A-filled.csv", filled_a), ("annotation-B-filled.csv", filled_b)):
        if not run.path("d2", name).exists():
            shutil.copyfile(path, run.path("d2", name))
    manifest = {"frozen_at": now(), "annotations_sha256": digest, "agreement": agreement,
                "annotators": sorted(set(a.annotator) | set(b.annotator)),
                "adjudicated_fields": 0 if adjudication is None else int(len(adjudication)),
                "final_counts": {f: final[f].value_counts().to_dict() for f in ("semantic_equivalent", "cached_answer_correct_for_target")},
                "by_split_role": final.groupby(["d2_split", "role"]).size().rename("n").reset_index().to_dict("records")}
    if adjudication_csv is not None and not run.path("d2", "adjudication.csv").exists():
        shutil.copyfile(adjudication_csv, run.path("d2", "adjudication.csv"))
    ce.exact_json(run.path("d2", "dataset-manifest.json"), manifest)
    return agreement, pending


def _d2_split_inputs(run, split, embedder, reranker, limit=16):
    """Cache = the split's originals + the D1 pool as background; queries = variants."""
    labels = ce.read_parquet(run.require("d2", "annotations.parquet"))
    items = labels[labels.d2_split.eq(split)].reset_index(drop=True)
    originals = run.d2_originals()
    d1_pool, pool = run.d1_pool()
    vectors = run.embeddings()
    index = {s: i for i, s in enumerate(pool.sample_id)}
    cache_frame = pd.concat([originals[originals.d2_split.eq(split)].drop(columns=["tokens", "stratum", "d2_split",
                                                                                     "negative_type", "group_id"], errors="ignore"),
                             d1_pool], ignore_index=True)
    cache_vectors = vectors[[index[s] for s in cache_frame.sample_id]]
    qpath = run.path("d2", f"{split}-variant-embeddings.npy")
    if qpath.exists():
        qvec = np.load(qpath)
    else:
        qvec = embedder.embed_all(items.variant_text.tolist(), progress=100)
        np.save(qpath, qvec)
    order, sims = ce.candidate_table(qvec, cache_vectors, limit)
    spath = run.path("d2", f"{split}-rerank-scores.npy")
    if spath.exists():
        scores = np.load(spath)
    else:
        texts = cache_frame.request_text.tolist()
        scores = np.stack([np.asarray(reranker.score(q, [texts[j] for j in row])) for q, row in zip(items.variant_text, order)])
        np.save(spath, scores)
    relation, status = ce.candidate_status(items, order, cache_frame, labels, pool)
    return items, labels, order, sims, scores, relation, status, cache_frame


def d2_calibrate(run, embedder, reranker):
    frozen = run.path("d2", "frozen-policy.json")
    if frozen.exists():
        raise FileExistsError("The policy is already frozen; calibration runs once per run directory")
    items, labels, order, sims, scores, relation, status, cache_frame = _d2_split_inputs(run, "calibration", embedder, reranker)
    spec, table = ce.calibrate(items, labels, order, sims, scores, relation, status, cache_frame, run.config)
    ce.write_parquet_once(run.path("d2", "calibration-grid.parquet"), table)
    production = {name: ce.d2_metrics(ce.evaluate_spec(s, items, labels, order, sims, scores, relation, status, cache_frame))
                  for name, s in ce.production_specs(run.config).items()}
    policy = {"frozen_at": now(), "spec": spec, "operational_policy": ce.calibrated_policy_spec(spec),
              "status": "feasible" if spec else "no_policy_meets_limit_fallback_exact_only",
              "limit": run.config["calibration"]["max_false_reuse_among_hits"],
              "models": run.config["models"], "annotations_sha256": ce.sha(run.path("d2", "annotations.parquet")),
              "calibration_grid_sha256": ce.sha(run.path("d2", "calibration-grid.parquet")),
              "calibration_production_metrics": production}
    ce.exact_json(frozen, policy)
    return policy, table


def d2_test(run, embedder, reranker):
    """The one test evaluation, after the policy freeze.

    Every attempt is appended to test-lock.json before any test item is read.
    Once test-metrics.json exists the test is spent and further runs are refused;
    an attempt that crashed before writing it stays on record and may be retried
    against the same frozen policy.
    """
    policy = ce.read(run.require("d2", "frozen-policy.json"))
    policy_sha = ce.sha(run.path("d2", "frozen-policy.json"))
    lock, done = run.path("d2", "test-lock.json"), run.path("d2", "test-metrics.json")
    if done.exists():
        raise FileExistsError("D2 test was already evaluated in this run; results are in d2/test-metrics.json")
    record = ce.read(lock) if lock.exists() else {"frozen_policy_sha256": policy_sha, "attempts": []}
    if record["frozen_policy_sha256"] != policy_sha:
        raise ValueError("frozen-policy.json changed after the first test attempt")
    record["attempts"].append({"started_at": now()})
    ce.write_json_atomic(lock, record)
    items, labels, order, sims, scores, relation, status, cache_frame = _d2_split_inputs(run, "test", embedder, reranker)
    specs = ce.production_specs(run.config)
    if policy["spec"] is not None:
        specs["calibrated"] = policy["spec"]
    outcomes, metrics = [], {}
    for name, spec in specs.items():
        o = ce.evaluate_spec(spec, items, labels, order, sims, scores, relation, status, cache_frame).assign(policy=name)
        outcomes.append(o)
        flagged = set(items.group_id[items.get("quality_flags", pd.Series("", index=items.index)).fillna("").ne("")])
        metrics[name] = {"spec": spec, "overall": ce.d2_metrics(o),
                         "excluding_flagged_groups": {"groups_excluded": sorted(flagged),
                                                      **ce.d2_metrics(o[~o.group_id.isin(flagged)])},
                         "by_stratum": {k: ce.d2_metrics(g) for k, g in o.groupby("stratum")},
                         "by_negative_type": {k: ce.d2_metrics(g) for k, g in o[o.role.eq("hard_negative")].groupby("negative_type")}}
    exact_hits = int(sum(ce.question_hash(t) in set(cache_frame.key_hash) for t in items.variant_text))
    metrics["C1"] = {"note": "exact-only; a variant hits only if identical after whitespace normalization", "exact_hits": exact_hits}
    ce.write_parquet_once(run.path("d2", "test-outcomes.parquet"), pd.concat(outcomes, ignore_index=True))
    record["attempts"][-1]["completed_at"] = now()
    ce.write_json_atomic(lock, record)
    ce.exact_json(done, metrics)
    return metrics


def run_d1_calibrated(run, reranker):
    policy = ce.read(run.require("d2", "frozen-policy.json"))
    calibrated = ce.policy_from("C_cal", policy["operational_policy"])
    policies = {**ce.config_policies(run.config), "C_cal": calibrated}
    return run_d1(run, reranker, arms=("C0", "C_cal"), policies=policies, tag="calibrated", capacities=False)


# ------------------------------------------------------------------ latency --

def latency_models(run, embedder, reranker, router=None):
    pool = run.pool()
    texts = pool.request_text.sample(frac=1.0, random_state=run.config["bootstrap"]["seed"]).tolist()
    import latency
    frame = latency.measure_models(texts, embedder, reranker, run.config, router.classify if router else None)
    ce.write_parquet_once(run.path("latency", "models.parquet"), frame)
    return frame


def router_spot_check(run, router, n=100):
    """Re-route a fixed sample on this machine (CPU) against prepare's routing (GPU on Colab)."""
    pool = run.pool().sample(n=n, random_state=run.config["bootstrap"]["seed"])
    local = [router.classify(t) for t in pool[run.config["router"]["request_input"]]]
    result = {"rows": n, "agreement": float((pd.Series(local, index=pool.index) == pool.routed_label).mean()),
              "device": str(router.device), "model_version": run.config["router"]["model_version"]}
    ce.exact_json(run.path("latency", "router-spot-check.json"), result)
    return result


def latency_pgvector(run, dsn):
    import latency
    pool, vectors = run.pool(), run.embeddings()
    frame = latency.measure_pgvector(dsn, pool, vectors, run.config)
    ce.write_parquet_once(run.path("latency", "pgvector.parquet"), frame)
    recall = [latency.hnsw_recall(dsn, pool, vectors, run.config, size) for size in run.config["latency"]["cache_sizes"]]
    ce.exact_json(run.path("latency", "hnsw-recall.json"), recall)
    ce.exact_json(run.path("latency", "container.json"), latency.container_info(run.config))
    return frame, recall


def pgvector_decision_parity(run, dsn, reranker, trace_id, arms=("C2", "C3")):
    """Replay one trace on real pgvector and compare with the in-memory replay."""
    import latency
    d1_pool, pool = run.d1_pool()
    vectors = run.embeddings()
    index = {s: i for i, s in enumerate(pool.sample_id)}
    vectors = vectors[[index[s] for s in d1_pool.sample_id]]
    trace = ce.read_parquet(run.require("d1", "traces", trace_id, "trace.parquet"))
    policies = ce.config_policies(run.config)
    scorer = ce.PairScorer(reranker.score, d1_pool.request_text.tolist())
    results = {}
    for arm in arms:
        cache = latency.PgCache(dsn, policies[arm].candidate_limit, run.config["latency"]["hnsw_ef_search"]).bind_pool(d1_pool)
        pg_log = ce.replay(trace, d1_pool, policies[arm], vectors, scorer, cache=cache)
        cache.close()
        memory_log = ce.read_parquet(run.require("d1", "traces", trace_id, arm, "requests.parquet"))
        summary, diff = latency.compare_decisions(memory_log, pg_log)
        ce.write_parquet_once(run.path("latency", f"pg-parity-{trace_id}-{arm}.parquet"), diff)
        results[arm] = summary
    ce.exact_json(run.path("latency", f"pg-parity-{trace_id}.json"), results)
    return results


# ------------------------------------------------------------------- report --

def bundle(run):
    """results-for-review.zip without raw benchmark text or embeddings."""
    target = run.path("report", "results-for-review.zip")
    skip = (".npy", "pool.parquet", "requests.parquet", "authoring", "annotation-", "adjudication", "variants.parquet",
            "annotations.parquet", "test-outcomes.parquet", "parity.parquet", "scorer-selfcheck.parquet")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.root.rglob("*")):
            if path.is_file() and path != target and not any(s in path.name for s in skip):
                z.write(path, path.relative_to(run.root))
    return target
