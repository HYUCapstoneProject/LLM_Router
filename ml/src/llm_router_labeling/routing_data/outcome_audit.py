"""Verified benchmark outcomes: audit recorded scores against public gold answers.

RouterBench and xRouteBench ship pre-graded model answers.  The 2026-10-01
audit found grading errors that changed which candidate "solved" a question:

* RouterBench GSM8K stores a 0.25-step rating, not correctness.  With the
  0.70 acceptance threshold, correct answers rated 0.5 were failures and
  GPT-4 looked worse than Mistral on 136 stream questions.
* MMLU's own answer key contains errors (MMLU-Redux 2.0).
* xRouteBench ``em_mc`` only accepts ``(X)``; a bare correct ``X`` scores 0.
  Its GSM8K/MATH graders miss equivalent formats (``$720.00``, ``{,}``,
  ``\\text{odd}``).  SQuAD token F1 >= 0.70 rejects verbose correct answers.

The audit writes per-cell corrections and item exclusions.  The dataset build
reads them through :func:`apply_routerbench_audit` and
:func:`apply_xroute_audit`; without an audit directory the legacy behaviour is
unchanged so frozen v1 artifacts stay reproducible.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import re
import string
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

AUDIT_VERSION = "benchmark-outcome-audit-v1"
MCQ_TASKS = ("mmlu", "hellaswag", "arc-challenge", "winogrande")
_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"
# MMLU-Redux error types that make the item itself unusable.
_REDUX_EXCLUDE = {
    "bad_question_clarity",
    "bad_options_clarity",
    "multiple_correct_answers",
    "no_correct_answer",
    "expert",
}


class OutcomeAuditError(RuntimeError):
    pass


# --------------------------------------------------------------------------- text helpers


def _unwrap(value: object) -> str:
    if isinstance(value, str) and value.startswith("["):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def _key_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("−", "-")
    return re.sub(r"[^0-9a-z]+", "", text)


def _option_key(value: object) -> str:
    """Lossless-enough key for telling options apart (keeps signs and symbols)."""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("−", "-")
    return re.sub(r"\s+", "", text)


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = text.replace(",", "").replace("$", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _last_boxed(text: object) -> str | None:
    text = str(text)
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    return None


def _clean_number_text(text: str) -> str:
    return text.replace("{,}", "").replace("\\,", "").replace("\\!", "").replace("\\$", "$")


def final_numbers(response: object) -> list[float]:
    """Candidate final answers of a free-form math response.

    A boxed or ``####`` answer wins.  Otherwise the first and the last number
    of the last line that contains a number are candidates: verbose answers
    restate the result first ("6 bottles ... for 180 days") while equations end
    with it ("3 + 4 = 7").  Numbers in the middle of the line never count.
    """
    text = _clean_number_text(_unwrap(response))
    boxed = _last_boxed(text)
    if boxed is not None:
        values = [_to_float(v) for v in re.findall(_NUM, boxed)]
        return [v for v in values if v is not None][-1:]
    marked = re.findall(r"####\s*(" + _NUM + ")", text)
    if marked:
        value = _to_float(marked[-1])
        return [] if value is None else [value]
    for line in reversed([line for line in re.split(r"\n+", text) if line.strip()]):
        values = [_to_float(v) for v in re.findall(_NUM, line)]
        values = [v for v in values if v is not None]
        if values:
            return list(dict.fromkeys([values[0], values[-1]]))
    return []


def strict_choice_letter(response: object) -> str | None:
    """Return a choice letter only when the response states it unambiguously."""
    text = _unwrap(response).strip()
    bare = re.compile(r"\**\(?([A-J])\)?\**\.?")
    match = bare.fullmatch(text)
    if match:
        return match.group(1)
    match = re.match(r"^\**\(?([A-J])[\.\):]\**\s+\S", text)
    if match:
        return match.group(1)
    stated = re.findall(
        r"(?:answer|choice)\s*(?:is|would be|:)?\s*[:：]?\s*\**\s*\(([A-J])\)", text, re.I
    )
    if stated:
        return stated[-1].upper()
    stated = re.findall(r"(?:answer|choice)\s*(?:is|:)\s*\**\s*([A-J])\b(?![\w'])", text)
    if stated:
        return stated[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        match = bare.fullmatch(lines[-1])
        if match:
            return match.group(1)
    return None


def _routerbench_letter(response: object) -> str | None:
    text = _unwrap(response).strip()
    match = re.match(r"^\(?([A-E])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"(?:answer|choice)\s*(?:is|:)?\s*\(?([A-E])\b", text, re.I)
    if match:
        return match.group(1).upper()
    letters = set(re.findall(r"\b([A-E])\)", text))
    return letters.pop() if len(letters) == 1 else None


def math_answer_key(text: str | None) -> str | None:
    if text is None:
        return None
    value = str(text)
    for old, new in (
        ("\\left", ""), ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
        ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("{,}", ""), ("$", ""),
        ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""),
    ):
        value = value.replace(old, new)
    value = re.sub(r"\\(?:text|mbox|mathrm)\{\s*([^}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", value)
    value = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", value)
    value = value.rstrip(".")
    if re.fullmatch(r"-?[\d,]+(?:\.\d+)?", value):
        number = float(value.replace(",", ""))
        value = str(int(number)) if number.is_integer() else repr(number)
    return value


def _squad_normalize(text: object) -> str:
    value = str(text).lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def squad_contains_gold(response: object, gold: object) -> bool:
    expected = _squad_normalize(gold)
    return bool(expected) and f" {expected} " in f" {_squad_normalize(response)} "


# --------------------------------------------------------------------------- gold loading


def _load_gold(gold_dir: Path, sources: Mapping[str, Any]) -> dict[str, Any]:
    for name, meta in sources.items():
        if "filename" not in meta:
            continue
        path = gold_dir / str(meta["filename"])
        if not path.is_file():
            raise OutcomeAuditError(f"Gold source is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            raise OutcomeAuditError(f"Gold source checksum mismatch: {name}")
    frame = lambda key: pd.read_parquet(gold_dir / str(sources[key]["filename"]))  # noqa: E731
    mcq: dict[tuple[str, str, frozenset[str]], set[str]] = collections.defaultdict(set)

    def add(task: str, stem: object, choices: Sequence[object], answer: int) -> None:
        options = [str(choice) for choice in choices]
        mcq[(task, _key_text(stem), frozenset(_key_text(o) for o in options))].add(
            _option_key(options[answer])
        )

    for key in ("mmlu_test", "mmlu_validation", "mmlu_dev"):
        for row in frame(key).itertuples():
            add("mmlu", row.question, list(row.choices), int(row.answer))
    for key in ("arc_test", "arc_validation", "arc_train"):
        for row in frame(key).itertuples():
            labels = list(row.choices["label"])
            add("arc-challenge", row.question, list(row.choices["text"]), labels.index(row.answerKey))
    for key in ("hellaswag_validation", "hellaswag_train"):
        for row in frame(key).itertuples():
            if str(row.label) != "":
                add("hellaswag", row.ctx, list(row.endings), int(row.label))
    for key in ("winogrande_validation", "winogrande_train"):
        for row in frame(key).itertuples():
            add("winogrande", row.sentence, [row.option1, row.option2], int(row.answer) - 1)
    gsm8k: dict[str, set[float]] = collections.defaultdict(set)
    for key in ("gsm8k_train", "gsm8k_test"):
        for row in frame(key).itertuples():
            value = _to_float(row.answer.split("####")[-1].strip())
            if value is not None:
                gsm8k[" ".join(row.question.split()).lower()].add(value)
    redux_frames = [
        pd.read_parquet(path).assign(subject=path.stem)
        for path in sorted((gold_dir / str(sources["mmlu_redux"]["directory"])).glob("*.parquet"))
    ]
    redux = pd.concat(redux_frames, ignore_index=True)
    digest = hashlib.sha256()
    for path in sorted((gold_dir / str(sources["mmlu_redux"]["directory"])).glob("*.parquet")):
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    if digest.hexdigest() != sources["mmlu_redux"]["files_sha256"]:
        raise OutcomeAuditError("MMLU-Redux files do not match the pinned release")
    if len(redux) != int(sources["mmlu_redux"]["rows"]):
        raise OutcomeAuditError("MMLU-Redux row count does not match the pinned release")
    return {"mcq": mcq, "gsm8k": gsm8k, "redux": redux}


def _redux_index(redux: pd.DataFrame) -> dict[tuple[str, frozenset[str]], dict[str, Any]]:
    index: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}
    for row in redux.itertuples():
        choices = [str(choice) for choice in row.choices]
        key = (_key_text(row.question), frozenset(_key_text(c) for c in choices))
        corrected: str | None = None
        if row.error_type == "wrong_groundtruth" and row.correct_answer is not None:
            value = str(row.correct_answer).strip()
            position = "ABCD".index(value) if value in "ABCD" else int(value) if value.isdigit() else None
            if position is not None and position < len(choices):
                corrected = _option_key(choices[position])
        index[key] = {"error_type": str(row.error_type), "corrected": corrected}
    return index


# --------------------------------------------------------------------------- RouterBench


def _mcq_target(prompt: object) -> tuple[str, list[tuple[str, str]]]:
    body = _unwrap(prompt)
    # Few-shot prompts end with the target block after the last solved example.
    body = re.split(r"Answer:\s*\n?\s*\(?[A-E]\)?\s*\n\n", body)[-1]
    options = re.findall(r"\n([A-E])\) (.*?)(?=\n[A-E]\) |\nPrint only|\Z)", body, flags=re.S)
    stem = re.split(r"\nA\) ", body, maxsplit=1)[0]
    stem = stem.replace("Please answer with the letter of the correct answer.", "")
    return stem, [(letter, text.strip()) for letter, text in options]


def _gsm8k_target(prompt: object) -> str:
    try:
        parts = ast.literal_eval(prompt) if isinstance(prompt, str) else list(prompt)
    except (ValueError, SyntaxError):
        parts = [str(prompt)]
    question = str(parts[-1])
    question = question.split("\n\nQuestion:")[-1]
    question = re.sub(r"^\s*Question:\s*", "", question)
    question = re.sub(r"\s*Answer:\s*$", "", question)
    return " ".join(question.split()).lower()


def audit_routerbench(
    labeled: pd.DataFrame,
    raw_by_shot: Mapping[str, pd.DataFrame],
    models: Sequence[str],
    gold: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (cell corrections, item exclusions, summary) for RouterBench."""
    redux = _redux_index(gold["redux"])
    corrections: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    summary: collections.Counter[str] = collections.Counter()
    raw_index = {
        shot: frame.drop_duplicates("sample_id").set_index("sample_id")
        for shot, frame in raw_by_shot.items()
    }
    for row in labeled.to_dict("records"):
        sample_id = str(row["sample_id"])
        task = re.sub(r"^mmlu-.*", "mmlu", str(row["eval_name"]))
        raw = raw_index[str(row.get("shot_source", "5shot"))]
        if sample_id not in raw.index:
            raise OutcomeAuditError(f"RouterBench raw response missing: {sample_id}")
        source = raw.loc[sample_id]
        recorded = {m: pd.to_numeric(row[m], errors="coerce") for m in models}
        if task == "grade-school-math":
            golds = gold["gsm8k"].get(_gsm8k_target(source["prompt"]))
            if not golds:
                exclusions.append({"sample_id": sample_id, "rule": "gsm8k_gold_unmatched"})
                summary["gsm8k_gold_unmatched"] += 1
                continue
            expected = next(iter(golds))
            for model in models:
                if pd.isna(recorded[model]):
                    continue
                solved = any(abs(v - expected) < 1e-6 for v in final_numbers(source[f"{model}|model_response"]))
                corrections.append(_cell(sample_id, model, recorded[model], float(solved), "gsm8k_numeric_regrade"))
                summary["gsm8k_cells_regraded"] += 1
                summary["gsm8k_success_flipped"] += int(solved != (recorded[model] >= 0.70))
            continue
        if task not in MCQ_TASKS:
            continue
        stem, options = _mcq_target(source["prompt"])
        option_keys = frozenset(_key_text(text) for _, text in options)
        answers = gold["mcq"].get((task, _key_text(stem), option_keys))
        if not answers or len(answers) != 1:
            exclusions.append({"sample_id": sample_id, "rule": "mcq_gold_unmatched_or_ambiguous"})
            summary["mcq_gold_unmatched_or_ambiguous"] += 1
            continue
        answer_key = next(iter(answers))
        rule = "mcq_gold_regrade"
        if task == "mmlu":
            review = redux.get((_key_text(stem), option_keys))
            if review is not None and review["error_type"] in _REDUX_EXCLUDE | {"wrong_groundtruth"}:
                if review["error_type"] == "wrong_groundtruth" and review["corrected"] is not None:
                    answer_key, rule = review["corrected"], "mmlu_redux_corrected_answer"
                    summary["mmlu_redux_corrected_items"] += 1
                else:
                    exclusions.append({"sample_id": sample_id, "rule": f"mmlu_redux_{review['error_type']}"})
                    summary[f"mmlu_redux_{review['error_type']}"] += 1
                    continue
        letters = [letter for letter, text in options if _option_key(text) == answer_key]
        if len(letters) != 1:
            exclusions.append({"sample_id": sample_id, "rule": "mcq_answer_not_in_options"})
            summary["mcq_answer_not_in_options"] += 1
            continue
        for model in models:
            if pd.isna(recorded[model]):
                continue
            letter = _routerbench_letter(source[f"{model}|model_response"])
            # Unparseable answers keep the recorded grade (RouterBench scored them 0).
            verified = recorded[model] if letter is None else float(letter == letters[0])
            if verified != recorded[model] or rule != "mcq_gold_regrade":
                corrections.append(_cell(sample_id, model, recorded[model], verified, rule))
                summary[f"{rule}_cells_changed"] += int(verified != recorded[model])
    corrections_frame = pd.DataFrame(
        corrections,
        columns=["sample_id", "model", "raw_performance", "verified_performance", "verified_accepted", "rule"],
    )
    exclusions_frame = pd.DataFrame(exclusions, columns=["sample_id", "rule"])
    return corrections_frame, exclusions_frame, dict(sorted(summary.items()))


def _cell(sample_id: str, model: str, raw: float, verified: float, rule: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "model": model,
        "raw_performance": float(raw),
        "verified_performance": float(verified),
        "verified_accepted": bool(verified >= 1.0),
        "rule": rule,
    }


def apply_routerbench_audit(
    accepted: pd.DataFrame,
    labeled: pd.DataFrame,
    audit: Mapping[str, pd.DataFrame],
    performance: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, set[str]]:
    """Overwrite accepted (and optionally performance) cells; return excluded ids.

    ``accepted`` is indexed like ``labeled`` (positional) with model columns.
    """
    corrections = audit["routerbench_corrections"]
    excluded = set(audit["routerbench_exclusions"]["sample_id"].astype(str))
    positions = pd.Series(np.arange(len(labeled)), index=labeled["sample_id"].astype(str))
    if positions.index.has_duplicates:
        raise OutcomeAuditError("RouterBench sample IDs must be unique to apply the audit")
    present = corrections["sample_id"].isin(positions.index) & corrections["model"].isin(accepted.columns)
    for row in corrections.loc[present].itertuples(index=False):
        position = positions[row.sample_id]
        accepted.iat[position, accepted.columns.get_loc(row.model)] = bool(row.verified_accepted)
        if performance is not None:
            performance.iat[position, performance.columns.get_loc(row.model)] = row.verified_performance
    return accepted, excluded


# --------------------------------------------------------------------------- xRouteBench


def xroute_cell_key(frame: pd.DataFrame) -> pd.Series:
    identifier = frame["task_id"].where(frame["task_id"].notna(), frame["embedding_id"]).astype(str)
    query_hash = frame["query"].astype(str).map(lambda q: hashlib.sha256(q.encode("utf-8")).hexdigest()[:24])
    return frame["task_name"].astype(str) + "|" + identifier + "|" + query_hash + "|" + frame["model_name"].astype(str)


def audit_xroute(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Correct grader false negatives; recorded successes are never revoked."""
    rows: list[dict[str, Any]] = []
    summary: collections.Counter[str] = collections.Counter()
    keys = xroute_cell_key(frame)
    for key, row in zip(keys, frame.itertuples(index=False)):
        recorded = float(row.performance) if pd.notna(row.performance) else np.nan
        if np.isnan(recorded):
            continue
        verified, rule = recorded, None
        if row.metric == "em_mc" and recorded == 0.0:
            if strict_choice_letter(row.response) == str(row.ground_truth).strip():
                verified, rule = 1.0, "em_mc_format_false_negative"
        elif row.metric == "GSM8K" and recorded == 0.0:
            truth = str(row.ground_truth)
            expected = _to_float((re.findall(_NUM, truth.split("####")[-1].replace(",", "")) or [None])[-1])
            if expected is not None and any(abs(v - expected) < 1e-6 for v in final_numbers(row.response)):
                verified, rule = 1.0, "gsm8k_format_false_negative"
        elif row.metric == "MATH" and recorded == 0.0:
            expected = math_answer_key(_last_boxed(row.ground_truth))
            given = math_answer_key(_last_boxed(row.response))
            if expected is not None and expected == given:
                verified, rule = 1.0, "math_equivalent_false_negative"
        elif row.metric == "f1_score":
            if recorded < 0.70 and squad_contains_gold(row.response, row.ground_truth):
                verified, rule = 1.0, "squad_contains_gold_answer"
        if rule is not None:
            rows.append({
                "cell_key": key,
                "task_name": str(row.task_name),
                "model_name": str(row.model_name),
                "raw_performance": recorded,
                "verified_performance": verified,
                "verified_accepted": True,
                "rule": rule,
            })
            summary[rule] += 1
    return pd.DataFrame(rows, columns=["cell_key", "task_name", "model_name", "raw_performance", "verified_performance", "verified_accepted", "rule"]), dict(sorted(summary.items()))


def apply_xroute_audit(output: pd.DataFrame, audit: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    corrections = audit["xroute_corrections"].set_index("cell_key")
    keys = xroute_cell_key(output)
    hit = keys.isin(corrections.index).to_numpy()
    if hit.any():
        matched = corrections.loc[keys[hit]]
        output.loc[hit, "performance"] = matched["verified_performance"].to_numpy(float)
        output.loc[hit, "accepted"] = matched["verified_accepted"].to_numpy(bool)
    output["outcome_verified"] = hit
    return output


# --------------------------------------------------------------------------- build / load


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_outcome_audit(config: Mapping[str, Any], *, project_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise OutcomeAuditError("Audit output directory must be new/empty")
    section = config["outcome_audit"]
    gold = _load_gold(project_root / str(section["gold_dir"]), section["gold_sources"])
    rb = config["routerbench"]
    labeled = pd.read_parquet(project_root / str(rb["labeled_path"]))
    tiers = json.loads((project_root / str(rb["model_tiers_path"])).read_text(encoding="utf-8"))
    models = [str(m) for tier in ("low", "medium", "high") for m in tiers["models"][tier]]
    raw_by_shot = {
        shot: pd.read_pickle(project_root / str(meta["local_path"]))
        for shot, meta in rb["raw_sources"].items()
    }
    for shot, meta in rb["raw_sources"].items():
        if _sha256(project_root / str(meta["local_path"])) != meta["sha256"]:
            raise OutcomeAuditError(f"RouterBench raw checksum mismatch: {shot}")
    rb_corrections, rb_exclusions, rb_summary = audit_routerbench(labeled, raw_by_shot, models, gold)

    xconfig = config["xroutebench"]
    xdir = project_root / "ml/data/raw/xroutebench" / str(xconfig["revision"])
    xframes = [pd.read_parquet(xdir / str(xconfig["sources"][split]["filename"])) for split in ("train", "test")]
    x_corrections, x_summary = audit_xroute(pd.concat(xframes, ignore_index=True))
    if x_corrections["cell_key"].duplicated().any():
        raise OutcomeAuditError("xRouteBench audit keys are not unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    rb_corrections.to_parquet(output_dir / "routerbench-corrections.parquet", index=False)
    rb_exclusions.to_parquet(output_dir / "routerbench-exclusions.parquet", index=False)
    x_corrections.to_parquet(output_dir / "xroutebench-corrections.parquet", index=False)
    summary = {
        "audit_version": AUDIT_VERSION,
        "routerbench": {
            **rb_summary,
            "cells_changed": int((rb_corrections["raw_performance"] != rb_corrections["verified_performance"]).sum()),
            "excluded_items": int(len(rb_exclusions)),
        },
        "xroutebench": {**x_summary, "cells_changed": int(len(x_corrections))},
    }
    (output_dir / "audit-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        name: _sha256(output_dir / name)
        for name in ("routerbench-corrections.parquet", "routerbench-exclusions.parquet", "xroutebench-corrections.parquet", "audit-summary.json")
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def load_outcome_audit(config: Mapping[str, Any], project_root: Path) -> dict[str, pd.DataFrame] | None:
    section = config.get("outcome_audit")
    if not section:
        return None
    directory = project_root / str(section["audit_dir"])
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        if _sha256(directory / name) != digest:
            raise OutcomeAuditError(f"Outcome audit artifact changed: {name}")
    expected = section.get("audit_manifest_sha256")
    if expected and _sha256(directory / "manifest.json") != expected:
        raise OutcomeAuditError("Outcome audit manifest does not match the pinned version")
    return {
        "routerbench_corrections": pd.read_parquet(directory / "routerbench-corrections.parquet"),
        "routerbench_exclusions": pd.read_parquet(directory / "routerbench-exclusions.parquet"),
        "xroute_corrections": pd.read_parquet(directory / "xroutebench-corrections.parquet"),
    }
