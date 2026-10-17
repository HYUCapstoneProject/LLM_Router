"""EV-10 length-cost sensitivity inputs (plan 2.3, sensitivity 1): input and output length of every dev/test question.

Local step (needs the raw benchmark files). For each dev and test question of the four sources:

  input_tokens          router (ModernBERT) tokenizer count of model_input_text, no truncation
  output_tokens_median  median over every data model with a valid answer of that model's output length
  output_models         how many models entered the median

Output lengths come from each source's own record: LLMRouterBench `completion_tokens`, SPROUT `num_output_tokens`,
xRouteBench `output_tokens`; RouterBench only keeps the answer text, so its answers are counted with the same router
tokenizer. Validity follows the external-data builder (a zero-cost, zero-token call is missing, not an answer).
No outcomes (success or score) are written. 07b and 08 turn these into per-question costs; the choices and the
reasons to check them later are in the plan, section 2.3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_track as C  # noqa: E402

ROUTERBENCH_LABELED = ("ml/artifacts/routerbench-labeled/withmartian-routerbench-2024-03-27__routerbench-capability-tier-v1"
                       "__overall-solve-rate-v3-probe-selected/labeled.parquet")
ROUTERBENCH_PICKLES = {"0shot": "ml/data/raw/routerbench_0shot.pkl", "5shot": "ml/data/raw/routerbench_5shot.pkl"}
LLMROUTERBENCH_ROOT = "ml/artifacts/external-eval-raw/llmrouterbench/bench-release"
SPROUT_DIR = "ml/artifacts/external-eval-raw/sprout"
TOKENIZER = "ml/artifacts/r0-release-20260927/approved"


def counter(tokenizer_dir: Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

    def count(texts) -> np.ndarray:
        texts = ["" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t) for t in texts]
        out = []
        for start in range(0, len(texts), 2048):
            out += [len(x) for x in tok(texts[start:start + 2048], add_special_tokens=True, truncation=False)["input_ids"]]
        return np.array(out, float)
    return count


def routerbench_outputs(root: Path, ids: set, count) -> pd.DataFrame:
    labeled = pd.read_parquet(root / ROUTERBENCH_LABELED, columns=["sample_id", "shot_source"])
    labeled = labeled[("routerbench:" + labeled.sample_id.astype(str)).isin(ids)]
    rows = []
    for shot, rel in ROUTERBENCH_PICKLES.items():
        want = labeled.loc[labeled.shot_source.eq(shot), "sample_id"].astype(str)
        if want.empty:
            continue
        raw = pd.read_pickle(root / rel)
        raw = raw[raw.sample_id.astype(str).isin(set(want))].drop_duplicates("sample_id")
        responses = [c for c in raw.columns if c.endswith("|model_response")]
        lengths = np.column_stack([np.where(raw[c].isna(), np.nan, count(raw[c].tolist())) for c in responses])
        lengths[lengths <= 2] = np.nan  # empty answer: only the tokenizer's special tokens
        rows.append(pd.DataFrame({"sample_id": "routerbench:" + raw.sample_id.astype(str).to_numpy(),
                                  "output_tokens_median": np.nanmedian(lengths, axis=1),
                                  "output_models": np.isfinite(lengths).sum(axis=1)}))
    return pd.concat(rows, ignore_index=True)


def xroutebench_outputs(root: Path, repo: Path, ids: set) -> pd.DataFrame:
    sys.path.insert(0, str(repo / "ml/src"))
    from llm_router_labeling.routing_data import preprocessing as builder
    import hashlib
    cfg = json.loads((repo / "ml/config/modernbert_initial_v2_verified_data.json").read_text())
    x = cfg["xroutebench"]
    data_dir = root / f"ml/data/raw/xroutebench/{x['revision']}"
    candidates = pd.read_parquet(data_dir / x["sources"]["candidates"]["filename"])["model_name"].astype(str).tolist()
    parts = []
    for split in ("train", "test"):
        raw = builder._validate_xroute_rows(pd.read_parquet(data_dir / x["sources"][split]["filename"]), split=split,
                                            candidates=candidates, exact_metrics=set(x["exact_metrics"]),
                                            continuous_metrics=set(x["continuous_metrics"]), structured=True)
        raw["sample_id"] = raw.query_key.map(lambda k: "xroutebench:" + hashlib.sha256(str(k).encode("utf-8")).hexdigest()[:20])
        raw = raw[raw.sample_id.isin(ids)]
        tokens = pd.to_numeric(raw["output_tokens"], errors="coerce").where(lambda v: v > 0)
        parts.append(raw.assign(tokens=tokens).groupby("sample_id").tokens.agg(["median", "count"]).reset_index())
    out = pd.concat(parts).drop_duplicates("sample_id")
    return out.rename(columns={"median": "output_tokens_median", "count": "output_models"})


def llmrouterbench_outputs(root: Path, repo: Path, ids: set) -> pd.DataFrame:
    cfg = json.loads((repo / "ml/config/external_evaluation_v1.json").read_text())["llmrouterbench"]
    base_root = root / LLMROUTERBENCH_ROOT
    rows = {}
    for dataset, spec in cfg["datasets"].items():
        base = base_root / dataset / spec["split"] if spec["split"] else base_root / dataset
        for model in cfg["pool_prices_usd_per_million_tokens"]:
            files = sorted((base / model).glob("*.json"))
            if len(files) != 1:
                raise SystemExit(f"expected one result file for {dataset}/{model}")
            for record in json.loads(files[0].read_text())["records"]:
                sid = f"llmrouterbench:{dataset}:{record['index']}"
                if sid not in ids:
                    continue
                completion, cost = record.get("completion_tokens"), record.get("cost")
                valid = record.get("score") is not None and ((cost or 0) > 0 or (completion or 0) > 0)
                rows.setdefault(sid, []).append(float(completion) if valid and completion else np.nan)
    return pd.DataFrame({"sample_id": list(rows), "output_tokens_median": [np.nanmedian(v) if np.isfinite(v).any() else np.nan
                                                                          for v in map(np.array, rows.values())],
                         "output_models": [int(np.isfinite(np.array(v)).sum()) for v in rows.values()]})


def sprout_outputs(root: Path, repo: Path, ids: set) -> pd.DataFrame:
    import pyarrow.parquet as pq
    cfg = json.loads((repo / "ml/config/external_evaluation_v1.json").read_text())["sprout"]
    frames = []
    for name, digest in cfg["files"].items():
        path = root / SPROUT_DIR / name
        if C.sha256_file(path) != digest:
            raise SystemExit(f"SPROUT file checksum differs: {name}")
        frames.append(pq.read_table(path, columns=["key", *cfg["pool"]]).to_pandas())
    data = pd.concat(frames, ignore_index=True)
    data["sample_id"] = "sprout:" + data.key.astype(str)
    data = data[data.sample_id.isin(ids)].drop_duplicates("sample_id")
    lengths = []
    for model in cfg["pool"]:
        values = data[model]
        score = values.map(lambda v: None if v is None else v.get("score"))
        tokens = values.map(lambda v: None if v is None else v.get("num_output_tokens")).astype(float)
        lengths.append(np.where(score.notna() & tokens.fillna(0).gt(0), tokens, np.nan))
    L = np.column_stack(lengths)
    return pd.DataFrame({"sample_id": data.sample_id.to_numpy(),
                         "output_tokens_median": np.nanmedian(np.where(np.isfinite(L), L, np.nan), axis=1),
                         "output_models": np.isfinite(L).sum(axis=1)})


def build(run_dir: Path, repo: Path, data_root: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise SystemExit("output directory must be new")
    prepare_dir = run_dir / "prepare"
    manifest = json.loads((prepare_dir / "manifest.json").read_text())
    frames = []
    for role in ("dev", "test"):
        path = prepare_dir / f"{role}.parquet"
        if C.sha256_file(path) != manifest["outputs"][path.name]["sha256"]:
            raise SystemExit(f"00_prepare output changed: {path.name}")
        frames.append(pd.read_parquet(path, columns=["sample_id", "source_dataset", "role", "model_input_text"]))
    q = pd.concat(frames, ignore_index=True)
    count = counter(data_root / TOKENIZER)
    q["input_tokens"] = count(q.model_input_text.tolist())
    by = {s: set(q.loc[q.source_dataset.eq(s), "sample_id"]) for s in C.SOURCES}
    outs = pd.concat([routerbench_outputs(data_root, by["routerbench"], count),
                      xroutebench_outputs(data_root, repo, by["xroutebench"]),
                      llmrouterbench_outputs(data_root, repo, by["llmrouterbench"]),
                      sprout_outputs(data_root, repo, by["sprout"])], ignore_index=True)
    if outs.sample_id.duplicated().any():
        raise SystemExit("duplicate output-length rows")
    table = q.drop(columns="model_input_text").merge(outs, on="sample_id", how="left")
    table["output_models"] = table.output_models.fillna(0).astype(int)
    output_dir.mkdir(parents=True)
    table.to_parquet(output_dir / "lengths.parquet", index=False)
    inputs = {"prepare_manifest": C.sha256_file(prepare_dir / "manifest.json"),
              "tokenizer.json": C.sha256_file(data_root / TOKENIZER / "tokenizer.json"),
              "routerbench_labeled": C.sha256_file(data_root / ROUTERBENCH_LABELED),
              **{f"routerbench_{k}": C.sha256_file(data_root / v) for k, v in ROUTERBENCH_PICKLES.items()},
              "llmrouterbench_archive_sha256_in_config": json.loads((repo / "ml/config/external_evaluation_v1.json")
                                                                    .read_text())["llmrouterbench"]["archive_sha256"]}
    summary = (table.groupby(["source_dataset", "role"])
               .agg(rows=("sample_id", "size"), missing_output=("output_tokens_median", lambda v: int(v.isna().sum())),
                    input_median=("input_tokens", "median"), output_median=("output_tokens_median", "median"),
                    models_median=("output_models", "median"))
               .reset_index().to_dict("records"))
    record = {"experiment": "EV-10 length-cost inputs (plan 2.3 sensitivity 1)", "inputs": inputs,
              "outputs": {"lengths.parquet": C.sha256_file(output_dir / "lengths.parquet")}, "summary": summary}
    (output_dir / "manifest.json").write_text(json.dumps(record, indent=1, default=float))
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None, help="checkout with the raw benchmark files (defaults to --repo)")
    p.add_argument("--output-dir", type=Path, required=True, help="normally <run>/length-inputs")
    a = p.parse_args(argv)
    print(json.dumps(build(a.run_dir, a.repo, a.data_root or a.repo, a.output_dir)["summary"], indent=1, default=float))


if __name__ == "__main__":
    main()
