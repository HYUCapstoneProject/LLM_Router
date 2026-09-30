"""Frozen external evaluation sets: LLMRouterBench (primary) and SPROUT (secondary).

Evaluation only. Recorded per-question outcomes of a fixed model pool are
converted to success with the project's rules, labelled with the project's
overall-solve-rate difficulty rule, de-duplicated against every question the
project already used, and split into calibration/test by duplicate group.
Nothing here calls a model API or trains anything.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..routing_data.text import compatible_near_duplicate, structured_duplicate_key
from ..shared.structured_text import clean_text
from ..training.routerbench_preprocessing import assign_duplicate_groups

LABELS = ("easy", "medium", "hard")


class ExternalDataError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def model_column(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


_MC_HEAD = re.compile(r"^Answer the following multiple choice question\..*?\n\n", re.DOTALL)
_MMLUPRO_HEAD = re.compile(r"^Answer the following [^\n]*? question\. The last line of your response[^\n]*\n\n")
_MMLUPRO_TAIL = re.compile(r"\n+Let's think step by step\.\s*$")
_BOXED_HEAD = re.compile(r"^Solve the following math problem step by step\.[^\n]*\n\n")
_BOXED_TAIL = re.compile(r"\n+Remember to put your final answer on the last line[^\n]*\s*$")
_HLE_HEAD = re.compile(r"^Your response should be in the following format:\n.*?\{your confidence score between 0% and 100% for your answer\}\.?\s*", re.DOTALL)
_LCB = re.compile(r"### Question:\n(.*?)\n*### Answer: \(use the provided format with backticks\)\s*$", re.DOTALL)


def strip_wrapper(kind: str, text: str) -> str | None:
    """Remove only the benchmark's known answer-format wrapper; None when it does not match."""
    text = str(text)
    if kind == "none":
        return clean_text(text)
    if kind == "multiple_choice_head":
        stripped, n = _MC_HEAD.subn("", text, count=1)
        return clean_text(stripped) if n else None
    if kind == "mmlupro":
        stripped, n = _MMLUPRO_HEAD.subn("", text, count=1)
        stripped, m = _MMLUPRO_TAIL.subn("", stripped, count=1)
        return clean_text(stripped) if n and m else None
    if kind == "boxed_math":
        stripped, n = _BOXED_HEAD.subn("", text, count=1)
        stripped, m = _BOXED_TAIL.subn("", stripped, count=1)
        return clean_text(stripped) if n and m else None
    if kind == "hle_head":
        stripped, n = _HLE_HEAD.subn("", text, count=1)
        return clean_text(stripped) if n else None
    if kind == "livecodebench":
        match = _LCB.search(text)
        return clean_text(match.group(1)) if match else None
    raise ExternalDataError(f"Unknown wrapper kind: {kind}")


_OPTION_LINE = re.compile(r"^[ \t]*(?:\(?[A-J][.):](?:\s|$)|Answer Choices:|Choices:|Options:)", re.MULTILINE)


def stem_key(text: str, minimum_words: int = 10) -> str | None:
    """Question text before the first option line, word-normalized.

    Catches the same question with shuffled or re-labelled options, which the
    structure-preserving near-duplicate guard deliberately refuses to merge.
    Short generic stems return None so they never merge unrelated questions.
    """
    cleaned = clean_text(text)
    match = _OPTION_LINE.search(cleaned)
    stem = cleaned[: match.start()] if match else cleaned
    words = re.findall(r"\w+", stem.casefold())
    return " ".join(words) if len(words) >= minimum_words else None


def passage_key(text: str, minimum_chars: int = 1000, prefix_chars: int = 300) -> str | None:
    """Shared long passage (e.g. one story with several questions): same split unit."""
    cleaned = " ".join(clean_text(text).split())
    return cleaned[:prefix_chars] if len(cleaned) >= minimum_chars else None


def labels_from_rates(rates: np.ndarray, hard: float, easy: float) -> np.ndarray:
    if not 0.0 <= hard < easy <= 1.0:
        raise ExternalDataError("Thresholds must satisfy 0 <= hard < easy <= 1")
    return np.where(rates >= easy, "easy", np.where(rates >= hard, "medium", "hard"))


def load_llmrouterbench(root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    """Wide table: one row per (dataset, index) with success/score/cost per pool model."""
    pool = list(config["pool_prices_usd_per_million_tokens"])
    rows = []
    for dataset, spec in config["datasets"].items():
        base = root / dataset / spec["split"] if spec["split"] else root / dataset
        records: dict[str, list[dict]] = {}
        for model in pool:
            files = sorted((base / model).glob("*.json"))
            if len(files) != 1:
                raise ExternalDataError(f"Expected one result file for {dataset}/{spec['split']}/{model}, found {len(files)}")
            records[model] = json.loads(files[0].read_text())["records"]
        by_index: dict[Any, dict[str, Any]] = {}
        for model, items in records.items():
            for record in items:
                entry = by_index.setdefault(record["index"], {"prompts": set(), "models": {}})
                entry["prompts"].add(record.get("prompt"))
                entry["models"][model] = record
        for index, entry in sorted(by_index.items(), key=lambda item: str(item[0])):
            prompt = next(iter(entry["prompts"])) if len(entry["prompts"]) == 1 else None
            row: dict[str, Any] = {
                "sample_id": f"llmrouterbench:{dataset}:{index}", "benchmark": "llmrouterbench",
                "source_family": dataset, "source_split": spec["split"], "slice": spec["slice"],
                "reference_prompt": prompt, "exclusion_reason": None}
            if prompt is None:
                row["exclusion_reason"] = "inconsistent_prompt_across_models"
            else:
                text = strip_wrapper(spec["wrapper"], prompt)
                if text is None:
                    row["exclusion_reason"] = "wrapper_mismatch"
                elif not text.strip():
                    row["exclusion_reason"] = "empty_after_wrapper"
                row["model_input_text"] = text
            for model in pool:
                column = model_column(model)
                record = entry["models"].get(model)
                score = None if record is None else record.get("score")
                cost = None if record is None else record.get("cost")
                completion = None if record is None else record.get("completion_tokens")
                # A zero-cost, zero-token record is a failed call: missing, never a wrong answer.
                valid = score is not None and ((cost or 0) > 0 or (completion or 0) > 0)
                row[f"valid_{column}"] = bool(valid)
                row[f"score_{column}"] = float(score) if valid else np.nan
                row[f"success_{column}"] = float(score >= 1.0) if valid else np.nan
                row[f"cost_{column}"] = float(cost) if valid else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def load_sprout(directory: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    pool = list(config["pool"])
    frames = []
    for name, expected in config["files"].items():
        path = directory / name
        if sha256_file(path) != expected:
            raise ExternalDataError(f"SPROUT file checksum differs: {name}")
        frames.append(pq.read_table(path, columns=["key", "dataset", "prompt", *pool]).to_pandas())
    data = pd.concat(frames, ignore_index=True)
    data = data[data.dataset.isin(config["datasets"])].reset_index(drop=True)
    if data.key.duplicated().any():
        raise ExternalDataError("SPROUT keys must be unique")
    out = pd.DataFrame({
        "sample_id": "sprout:" + data.key.astype(str), "benchmark": "sprout",
        "source_family": data.dataset, "source_split": "all",
        "slice": data.dataset.map(lambda d: config["datasets"][d]["slice"]),
        "reference_prompt": data.prompt, "model_input_text": data.prompt.map(clean_text)})
    out["exclusion_reason"] = np.where(out.model_input_text.str.strip().eq(""), "empty_after_wrapper", None)
    threshold = float(config["success_threshold"])
    for model in pool:
        column = model_column(model)
        values = data[model]
        score = values.map(lambda x: None if x is None else x.get("score"))
        tokens = values.map(lambda x: None if x is None else x.get("num_output_tokens"))
        valid = score.notna() & tokens.fillna(0).gt(0)
        out[f"valid_{column}"] = valid.to_numpy()
        out[f"score_{column}"] = np.where(valid, score.astype(float), np.nan)
        out[f"success_{column}"] = np.where(valid, (score.astype(float) >= threshold).astype(float), np.nan)
    return out


def add_labels(frame: pd.DataFrame, pool_columns: list[str], labels: Mapping[str, Any], minimum_valid: int) -> pd.DataFrame:
    frame = frame.copy()
    success = frame[[f"success_{c}" for c in pool_columns]].to_numpy(float)
    frame["valid_model_count"] = np.isfinite(success).sum(1)
    frame["accepted_model_count"] = np.nansum(success, 1).astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = frame.accepted_model_count / frame.valid_model_count
    frame["overall_solve_rate"] = rates
    few = frame.valid_model_count.lt(minimum_valid) & frame.exclusion_reason.isna()
    frame.loc[few, "exclusion_reason"] = "too_few_valid_models"
    frame["difficulty_label"] = labels_from_rates(rates.fillna(0).to_numpy(), float(labels["hard"]), float(labels["easy"]))
    frame.loc[frame.valid_model_count.lt(minimum_valid), "difficulty_label"] = None
    return frame


def mark_overlaps(external: pd.DataFrame, reference: pd.DataFrame, dedup: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Group external candidates with the project's questions using the project's duplicate rule.

    External rows sharing a group with any project question are excluded.
    A SPROUT row sharing a group with an LLMRouterBench row is excluded in
    favour of the primary (cost-bearing) copy. The group id is kept for the
    calibration/test split so duplicates never straddle it.
    """
    external = external.copy()
    live = external.exclusion_reason.isna()
    combined = pd.concat([
        reference[["sample_id", "model_input_text"]].assign(_kind="reference", _benchmark="reference"),
        external.loc[live, ["sample_id", "model_input_text"]].assign(_kind="external", _benchmark=external.loc[live, "benchmark"]),
    ], ignore_index=True)
    groups, metrics = assign_duplicate_groups(
        combined.model_input_text,
        jaccard_threshold=float(dedup["near_duplicate_jaccard_threshold"]),
        minimum_tokens=int(dedup["near_duplicate_min_tokens"]),
        maximum_bucket_size=int(dedup["near_duplicate_max_bucket_size"]),
        normalizer=structured_duplicate_key, near_duplicate_guard=compatible_near_duplicate)
    combined["duplicate_group_id"] = groups.to_numpy()
    combined["stem_key"] = combined.model_input_text.map(stem_key)
    reference_groups = set(combined.loc[combined._kind.eq("reference"), "duplicate_group_id"])
    reference_stems = set(combined.loc[combined._kind.eq("reference"), "stem_key"].dropna())
    primary = combined._benchmark.eq("llmrouterbench")
    primary_groups = set(combined.loc[primary, "duplicate_group_id"])
    primary_stems = set(combined.loc[primary, "stem_key"].dropna())
    ext = combined[combined._kind.eq("external")].set_index("sample_id")
    external["duplicate_group_id"] = external.sample_id.map(ext.duplicate_group_id)
    external["stem_key"] = external.sample_id.map(ext.stem_key)
    by_group = external.duplicate_group_id.isin(reference_groups) & live
    by_stem = external.stem_key.isin(reference_stems) & live & ~by_group
    external.loc[by_group | by_stem, "exclusion_reason"] = "overlaps_project_data"
    secondary = (external.benchmark.eq("sprout") & external.exclusion_reason.isna()
                 & (external.duplicate_group_id.isin(primary_groups) | external.stem_key.isin(primary_stems)))
    external.loc[secondary, "exclusion_reason"] = "duplicate_of_primary_external"
    kept = external.exclusion_reason.isna()
    external["split_group_id"] = None
    external.loc[kept, "split_group_id"] = split_groups(external[kept])
    counts = lambda mask: external[mask].groupby(["benchmark", "source_family"]).size().rename("rows").reset_index().to_dict("records")
    report = {"duplicate_detection": metrics,
              "overlap_rows_by_family_exact_or_near": counts(by_group),
              "overlap_rows_by_family_same_stem": counts(by_stem),
              "secondary_duplicate_rows": int(secondary.sum())}
    return external.drop(columns="stem_key"), report


def split_groups(frame: pd.DataFrame) -> pd.Series:
    """Union duplicate groups with shared stems and shared long passages."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group, stem, text in zip(frame.duplicate_group_id.astype(str), frame.stem_key, frame.model_input_text):
        root = find("g:" + group)
        for key in (None if stem is None or (isinstance(stem, float) and np.isnan(stem)) else "s:" + stem,
                    None if not isinstance(text, str) or passage_key(text) is None else "p:" + passage_key(text)):
            if key is not None:
                parent[find(key)] = root
    return frame.duplicate_group_id.astype(str).map(lambda g: find("g:" + g))


def assign_split(frame: pd.DataFrame, fraction: float, seed: int) -> pd.Series:
    def stable(value: str) -> float:
        return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big") / 2**64
    return frame.split_group_id.astype(str).map(lambda g: "calibration" if stable(g) < fraction else "test")


def token_counts(texts: pd.Series, tokenizer_dir: Path | None) -> pd.Series:
    if tokenizer_dir is None:
        return pd.Series(np.nan, index=texts.index)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    ids = tokenizer(texts.fillna("").tolist(), add_special_tokens=True, truncation=False)["input_ids"]
    return pd.Series([len(x) for x in ids], index=texts.index)


def prepare(config_path: Path, repo_root: Path, lrb_root: Path, sprout_dir: Path, output_dir: Path,
            tokenizer_dir: Path | None = None) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text())
    if output_dir.exists():
        raise ExternalDataError("Use a new output directory; prepared sets are frozen")
    reference_files = {}
    references = []
    for relative in config["reference_data"]["files"]:
        path = repo_root / relative
        reference_files[relative] = sha256_file(path)
        references.append(pd.read_parquet(path, columns=["sample_id", "model_input_text"]).assign(sample_id=lambda f: relative + ":" + f.sample_id.astype(str)))
    reference = pd.concat(references, ignore_index=True)
    lrb_cfg, sprout_cfg = config["llmrouterbench"], config["sprout"]
    lrb_pool = [model_column(m) for m in lrb_cfg["pool_prices_usd_per_million_tokens"]]
    sprout_pool = [model_column(m) for m in sprout_cfg["pool"]]
    lrb = add_labels(load_llmrouterbench(lrb_root, lrb_cfg), lrb_pool, config["labels"], int(lrb_cfg["minimum_valid_models"]))
    sprout = add_labels(load_sprout(sprout_dir, sprout_cfg), sprout_pool, config["labels"], int(sprout_cfg["minimum_valid_models"]))
    external = pd.concat([lrb, sprout], ignore_index=True)
    external, overlap_report = mark_overlaps(external, reference, config["deduplication"])
    kept = external.exclusion_reason.isna()
    external["eval_split"] = None
    external.loc[kept, "eval_split"] = assign_split(external[kept], float(config["split"]["calibration_fraction"]), int(config["split"]["seed"]))
    external["input_tokens"] = np.nan
    external.loc[kept, "input_tokens"] = token_counts(external.loc[kept, "model_input_text"], tokenizer_dir)
    output_dir.mkdir(parents=True)
    outputs = {}
    for benchmark, pool in (("llmrouterbench", lrb_pool), ("sprout", sprout_pool)):
        part = external[external.benchmark.eq(benchmark)]
        columns = [c for c in part.columns if part[c].notna().any() or c in ("exclusion_reason", "eval_split")]
        prepared = part[part.exclusion_reason.isna()][columns].reset_index(drop=True)
        path = output_dir / f"{benchmark}.parquet"
        prepared.to_parquet(path, index=False)
        outputs[benchmark] = {"file": path.name, "sha256": sha256_file(path), "rows": len(prepared)}
    exclusions = external.loc[~kept, ["sample_id", "benchmark", "source_family", "exclusion_reason"]]
    exclusions.to_csv(output_dir / "exclusions.csv", index=False)
    summary = summarize(external, config, reference_files, outputs, overlap_report, tokenizer_dir)
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n")
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value))


def summarize(external: pd.DataFrame, config: Mapping[str, Any], reference_files: Mapping[str, str],
              outputs: Mapping[str, Any], overlap_report: Mapping[str, Any], tokenizer_dir: Path | None) -> dict[str, Any]:
    kept = external[external.exclusion_reason.isna()]
    def table(frame: pd.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
        return frame.groupby(keys, dropna=False).size().rename("rows").reset_index().to_dict("records")
    lrb = kept[kept.benchmark.eq("llmrouterbench")]
    costs = {c[len("cost_"):]: float(lrb[c].mean()) for c in lrb.columns if c.startswith("cost_") and lrb[c].notna().any()}
    success = {c[len("success_"):]: {"benchmark": b, "mean": float(g[c].mean())}
               for b, g in kept.groupby("benchmark") for c in g.columns if c.startswith("success_") and g[c].notna().any()}
    return {
        "version": config["version"],
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "sources": {"llmrouterbench": {k: config["llmrouterbench"][k] for k in ("repository", "revision", "archive_sha256", "license_status")},
                    "sprout": {k: config["sprout"][k] for k in ("repository", "revision", "files", "license_status", "cost_status")}},
        "reference_files_sha256": reference_files,
        "outputs": outputs,
        "rows_before_exclusion": table(external, ["benchmark", "source_family"]),
        "exclusions": table(external[external.exclusion_reason.notna()], ["benchmark", "source_family", "exclusion_reason"]),
        "prepared_by_slice_split_label": table(kept, ["benchmark", "slice", "eval_split", "difficulty_label"]),
        "prepared_by_family": table(kept, ["benchmark", "slice", "source_family"]),
        "overlap": overlap_report,
        "input_tokens": {"tokenizer": str(tokenizer_dir) if tokenizer_dir else None,
                         "rows_over_8192": int(kept.input_tokens.gt(8192).sum()) if kept.input_tokens.notna().any() else None,
                         "median": float(kept.input_tokens.median()) if kept.input_tokens.notna().any() else None,
                         "max": float(kept.input_tokens.max()) if kept.input_tokens.notna().any() else None},
        "mean_recorded_cost_usd_llmrouterbench": costs,
        "mean_success_by_model": success,
        "evaluation_only": True,
    }
