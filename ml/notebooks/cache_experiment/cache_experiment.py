"""Cache experiment harness (C0/C1/C2/C3). Local only.

The contract is docs/ml/CACHE_EXPERIMENT.md. Recorded benchmark answers are
replayed through a re-implementation of the operational Spring cache
(CacheService + RerankGateService + QuestionHash) so that exact, semantic and
reranked reuse can be compared on the same request order. Nothing here calls an
LLM, writes to the operational database or changes operational settings.
Model inference (BGE, cross-encoder) lives in models.py; pgvector and timing
live in latency.py. This module is pure Python/numpy so it can be unit tested.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("easy", "medium", "hard")
TIERS = ("low", "medium", "high")
LABEL_TO_TIER = dict(zip(LABELS, TIERS))
ARMS = ("C0", "C1", "C2", "C3")
REUSE_RECORDED, REUSE_CORRECT, REUSE_INCORRECT, REUSE_UNKNOWN = "recorded", "correct", "incorrect", "unknown"
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- artifacts --

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


VOLATILE_KEYS = ("created_at", "frozen_at", "locked_at")


def _stable(value):
    return {k: v for k, v in value.items() if k not in VOLATILE_KEYS} if isinstance(value, dict) else value


def jsonable(value):
    """Plain JSON types; NaN/inf become null (a ratio with an empty denominator)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def exact_json(path, value):
    """Write-once JSON. Re-running a stage may only reproduce the same content.

    Top-level timestamps are ignored in the comparison so an interrupted notebook
    can be re-run; one-shot stages (policy freeze, test lock) guard themselves.
    The write is atomic, so a crash never leaves a half-written seal.
    """
    path = Path(path)
    value = jsonable(value)
    if path.exists():
        if _stable(read(path)) != _stable(value):
            raise ValueError(f"Existing artifact differs; use a new run directory: {path}")
        return
    write_json_atomic(path, value)


def write_parquet_once(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        probe = path.with_name(f".{path.name}.probe")
        frame.reset_index(drop=True).to_parquet(probe, index=False)
        same = sha(probe) == sha(path)
        probe.unlink()
        if not same:
            raise ValueError(f"Existing parquet differs; use a new run directory: {path}")
        return sha(path)
    frame.reset_index(drop=True).to_parquet(path, index=False)
    return sha(path)


def read_parquet(path):
    return pd.read_parquet(path).reset_index(drop=True)


def file_pin(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Real SHA-256 required for {path}")
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path} expected {expected} got {actual}")


def load_config(path=HERE / "experiment_config.json"):
    config = read(path)
    if config.get("schema_version") != "cache-experiment-v1":
        raise ValueError("Unsupported cache experiment config")
    for name in ARMS:
        if name not in config["policies"]:
            raise ValueError(f"Policy {name} is missing")
    return config


def input_path(config, name, data_root):
    spec = config["inputs"][name]
    path = Path(data_root) / spec["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Input {name} not found at {path}")
    file_pin(path, spec["sha256"])
    return path


# ------------------------------------------------ operational exact key -----
# Mirrors backend QuestionHash.normalize/of. Java Character.isWhitespace is not
# Python str.isspace: NBSP-like separators and U+0085 are excluded, and the
# check runs per UTF-16 code unit. Parity is pinned by a Java-generated fixture.

_JAVA_SPACE_EXCLUDED = {0x00A0, 0x2007, 0x202F}


def java_is_whitespace(code_unit):
    if 0x09 <= code_unit <= 0x0D or 0x1C <= code_unit <= 0x1F:
        return True
    if code_unit in _JAVA_SPACE_EXCLUDED:
        return False
    import unicodedata
    return unicodedata.category(chr(code_unit)) in ("Zs", "Zl", "Zp")


def _utf16_units(text):
    raw = text.encode("utf-16-be", "surrogatepass")
    return [int.from_bytes(raw[i:i + 2], "big") for i in range(0, len(raw), 2)]


def normalize_question(question):
    if question is None:
        return ""
    units, out, pending = _utf16_units(str(question)), [], False
    for unit in units:
        if java_is_whitespace(unit):
            pending = bool(out)
        else:
            if pending:
                out.append(0x20)
                pending = False
            out.append(unit)
    return b"".join(u.to_bytes(2, "big") for u in out).decode("utf-16-be", "surrogatepass")


def question_hash(question):
    # Java getBytes(UTF_8) replaces an unpaired surrogate with '?'.
    data = normalize_question(question).encode("utf-8", "replace")
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------ answer text ---

def response_text(value):
    """RouterBench stores responses as the repr of a list of turns."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            turns = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(turns, list) and all(isinstance(t, str) for t in turns):
            return "\n\n".join(turns)
    return text


_RB_OPTION = re.compile(r"^([A-J])\) ?(.*\S)\s*$", re.MULTILINE)
_XRB_OPTION = re.compile(r"^\(?([A-J])(?:\)|\.)\s+(.*\S)\s*$", re.MULTILINE)
_RB_MCQ_TASK = re.compile(r"^(mmlu-.*|arc-challenge|hellaswag|winogrande)$")


def parse_options(text, source):
    pattern = _RB_OPTION if source == "routerbench" else _XRB_OPTION
    options = {}
    for letter, body in pattern.findall(str(text)):
        if letter in options:
            return {}
        options[letter] = option_text_key(body)
    letters = sorted(options)
    if not letters or letters != [chr(ord("A") + i) for i in range(len(letters))]:
        return {}
    return options


def option_text_key(body):
    return " ".join(str(body).casefold().split()).rstrip(".")


def rb_letter(text):
    """Letter chosen by a RouterBench 'Print only a single choice' answer.

    Returns (letter, status). status is 'ok', 'none' (no option-like token at
    all) or 'ambiguous'. Only unambiguous leading letters or an explicit
    'answer is X' are accepted.
    """
    s = str(text).strip()
    m = re.fullmatch(r"\(?([A-J])\)?[.):]?", s)
    if m:
        return m.group(1), "ok"
    # A leading letter counts only with closing punctuation: "B) because" yes,
    # "I think" or "A cat" no.
    m = re.match(r"(?:\(([A-J])\)|([A-J])[.):])(?:\s|$)", s)
    if m:
        return m.group(1) or m.group(2), "ok"
    explicit = set(re.findall(r"(?i:answer|choice|option)\s*(?:is|:)?\s*\(?([A-J])\)?(?![A-Za-z])", s))
    if len(explicit) == 1:
        return explicit.pop(), "ok"
    if explicit or re.search(r"(?<![A-Za-z])\(?[A-J]\)(?![A-Za-z])", s):
        return None, "ambiguous"
    return None, "none"


def xrb_letter(text):
    letters = re.findall(r"\(([A-J])\)", str(text))
    if not letters:
        return None, "none"
    if len(set(letters)) > 1:
        return None, "ambiguous"
    return letters[-1], "ok"


def choice_letter(text):
    """Option letter of an answer in either benchmark's format."""
    letter, status = xrb_letter(text)
    return (letter, status) if status != "none" else rb_letter(text)


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_boxed(text):
    s, key, out = str(text), "\\boxed{", None
    start = s.rfind(key)
    if start < 0:
        return None
    depth, i = 1, start + len(key)
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                out = s[i:j]
                break
    return out


def canonical_answer(value):
    if value is None:
        return None
    s = str(value).strip().strip("$").replace("\\!", "").replace("\\,", "")
    s = re.sub(r"\\(?:left|right|displaystyle)", "", s).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = "".join(s.split()).rstrip(".")
    return s or None


def as_number(value):
    s = canonical_answer(value)
    if s is None:
        return None
    s = s.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"-?\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and float(m.group(2)) != 0:
        sign = -1.0 if s.startswith("-") else 1.0
        return sign * float(m.group(1)) / float(m.group(2))
    return None


def numeric_answer(text):
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    m = re.findall(r"####\s*(" + _NUMBER.pattern + ")", str(text))
    if m:
        return m[-1]
    numbers = _NUMBER.findall(str(text))
    return numbers[-1] if numbers else None


def xrb_gold(metric, ground_truth):
    if metric == "em_mc":
        gt = str(ground_truth).strip()
        return ("mcq", gt) if re.fullmatch(r"[A-J]", gt) else ("unscorable", None)
    if metric in ("GSM8K", "MATH"):
        value = numeric_answer(ground_truth) if metric == "GSM8K" else last_boxed(ground_truth)
        if value is None and metric == "MATH":
            value = str(ground_truth).strip()
        return ("numeric", canonical_answer(value)) if value is not None else ("unscorable", None)
    return "unscorable", None


def rb_consensus_gold(row, models):
    """Gold option for a RouterBench MCQ from the models RouterBench scored 1.0.

    RouterBench stores per-model scores but not the gold option. For exact-match
    MCQ tasks a 1.0 means the model's choice was the gold option, so the gold is
    recovered only when every parsable correct response names the same letter.
    """
    letters = set()
    for model in models:
        if float(row.get(model, float("nan"))) == 1.0:
            letter, status = rb_letter(response_text(row.get(f"{model}|model_response")))
            if status == "ok":
                letters.add(letter)
    return letters.pop() if len(letters) == 1 else None


# ----------------------------------------------------------- pool building --

def build_pool(stream, logger, routerbench_raw, xroute_raw, config):
    """One row per stream question with the routed tier's recorded answer.

    The router is fixed: `logger` holds the operational 1.0 model's label per
    sample_id (pipeline.route_requests routes the request text). The request text is the stream query_text (the user question the
    router saw). RouterBench outcomes in the stream come from the 5-shot run, so
    `routerbench_raw` must be routerbench_5shot.pkl: its recorded responses are
    the answers whose scores and costs the stream carries (checked row by row).
    Every exclusion is returned with its reason.
    """
    tier_models = config["tier_models"]
    frame = stream.merge(logger[["sample_id", "initial_label"]], on="sample_id", how="left", validate="one_to_one")
    if frame.initial_label.isna().any():
        raise ValueError("Every stream row needs a logger prediction")
    frame["routed_label"] = frame.initial_label
    frame["routed_tier"] = frame.initial_label.map(LABEL_TO_TIER)
    parts, exclusions = [], []

    rb = frame[frame.source_dataset.eq("routerbench")].copy()
    if len(rb):
        raw = routerbench_raw.set_index("sample_id")
        ids = rb.sample_id.str.removeprefix("routerbench:")
        missing = sorted(set(ids) - set(raw.index))
        if missing:
            raise ValueError(f"RouterBench raw rows missing for {len(missing)} stream rows: {missing[:3]}")
        raw = raw.loc[ids.to_numpy()]
        all_models = [c for c in raw.columns if "|" not in c and c not in ("prompt", "eval_name", "oracle_model_to_route_to")]
        for tier in TIERS:
            model = tier_models["routerbench"][tier]
            if not rb[f"model_{tier}"].eq(model).all():
                raise ValueError(f"RouterBench {tier} model differs from the pinned tier model")
            rb[f"answer_{tier}"] = [response_text(v) for v in raw[f"{model}|model_response"]]
            score = pd.to_numeric(raw[model], errors="coerce").to_numpy()
            if not np.allclose(score, rb[f"quality_{tier}"].to_numpy(float), atol=1e-9, equal_nan=False):
                raise ValueError(f"RouterBench {tier} recorded score differs from the stream quality; use the 5-shot run")
            cost = pd.to_numeric(raw[f"{model}|total_cost"], errors="coerce").to_numpy()
            if not np.allclose(cost, rb[f"actual_cost_{tier}"].to_numpy(float), rtol=1e-9, atol=0):
                raise ValueError(f"RouterBench {tier} recorded cost differs from the stream cost")
            rb[f"latency_s_{tier}"] = np.nan
        rb["task_family"] = np.where(rb.source_task.str.match(_RB_MCQ_TASK), "mcq", "unscorable")
        golds = [rb_consensus_gold(r, all_models) if fam == "mcq" else None
                 for fam, (_, r) in zip(rb.task_family, raw.reset_index().iterrows())]
        rb["gold"] = golds
        rb["options_json"] = [json.dumps(parse_options(t, "routerbench"), sort_keys=True) for t in rb.query_text]
        rb["gold_origin"] = np.where(rb.gold.notna(), "routerbench_consensus_of_correct_models", "none")
        parts.append(rb)

    xrb = frame[frame.source_dataset.eq("xroutebench")].copy()
    if len(xrb):
        raw = xroute_raw.copy()
        raw["source_id"] = raw.task_id.where(raw.task_id.notna(), raw.embedding_id).astype(str)
        raw = raw[raw.model_name.isin(set(tier_models["xroutebench"].values()))]
        key = ["task_name", "source_id", "query"]
        if raw.duplicated(key + ["model_name"]).any():
            raise ValueError("xRouteBench has duplicate query/model runs")
        for tier in TIERS:
            model = tier_models["xroutebench"][tier]
            if not xrb[f"model_{tier}"].eq(model).all():
                raise ValueError(f"xRouteBench {tier} model differs from the pinned tier model")
            runs = raw[raw.model_name.eq(model)].rename(columns={
                "response": f"answer_{tier}", "performance": f"_perf_{tier}", "response_time": f"latency_s_{tier}",
                "ground_truth": f"_gt_{tier}", "metric": f"_metric_{tier}"})
            xrb = xrb.merge(runs[key + [f"answer_{tier}", f"_perf_{tier}", f"latency_s_{tier}", f"_gt_{tier}", f"_metric_{tier}"]],
                            left_on=["category", "source_task", "query_text"], right_on=key, how="left",
                            validate="one_to_one").drop(columns=key)
            if xrb[f"_perf_{tier}"].isna().any():
                raise ValueError(f"xRouteBench {tier} run missing for some stream rows")
            if not np.allclose(xrb[f"_perf_{tier}"].to_numpy(float), xrb[f"quality_{tier}"].to_numpy(float), atol=1e-9):
                raise ValueError(f"xRouteBench {tier} recorded performance differs from the stream quality")
            xrb[f"answer_{tier}"] = xrb[f"answer_{tier}"].fillna("").astype(str)
        if not (xrb._gt_low.astype(str).eq(xrb._gt_high.astype(str)) & xrb._metric_low.eq(xrb._metric_high)).all():
            raise ValueError("xRouteBench ground truth differs between tier runs")
        fam_gold = [xrb_gold(m, g) for m, g in zip(xrb._metric_low, xrb._gt_low)]
        xrb["task_family"] = [f for f, _ in fam_gold]
        xrb["gold"] = [g for _, g in fam_gold]
        xrb["options_json"] = [json.dumps(parse_options(t, "xroutebench"), sort_keys=True) for t in xrb.query_text]
        xrb["gold_origin"] = np.where(xrb.gold.notna(), "xroutebench_ground_truth", "none")
        xrb = xrb.drop(columns=[c for c in xrb.columns if c.startswith("_")])
        parts.append(xrb)

    pool = pd.concat(parts, ignore_index=True)
    pool["request_text"] = pool.query_text.astype(str)
    pool["key_hash"] = pool.request_text.map(question_hash)
    for column in ("answer", "quality", "success", "actual_cost", "latency_s"):
        pool[f"routed_{column}"] = [row[f"{column}_{tier}"] for tier, (_, row) in zip(pool.routed_tier, pool.iterrows())]
    pool.loc[pool.task_family.eq("mcq") & pool.gold.isna(), "task_family"] = "mcq_no_gold"
    duplicated = pool.key_hash.duplicated(keep="first")
    if duplicated.any():
        exclusions += [{"sample_id": s, "reason": "duplicate_exact_key"} for s in pool.sample_id[duplicated]]
        pool = pool[~duplicated]
    for column in ("routed_quality", "routed_actual_cost"):
        bad = ~np.isfinite(pool[column].to_numpy(float))
        if bad.any():
            exclusions += [{"sample_id": s, "reason": f"nonfinite_{column}"} for s in pool.sample_id[bad]]
            pool = pool[~bad]
    return pool.sort_values("sample_id", kind="stable").reset_index(drop=True), pd.DataFrame(exclusions, columns=["sample_id", "reason"])


def pool_summary(pool):
    return {
        "rows": int(len(pool)),
        "by_source": pool.source_dataset.value_counts().sort_index().to_dict(),
        "by_family": pool.task_family.value_counts().sort_index().to_dict(),
        "routed_tier": pool.routed_tier.value_counts().sort_index().to_dict(),
        "empty_routed_answers": int(pool.routed_answer.fillna("").str.strip().eq("").sum()),
        "mean_routed_quality": float(pool.routed_quality.mean()),
        "total_routed_cost": float(pool.routed_actual_cost.sum()),
    }


# ------------------------------------------------------- answer-reuse score --

def reuse_outcome(entry, target):
    """Is the answer stored for `entry` correct for the question `target`?

    Same question group -> the recorded score applies. Otherwise the answer is
    re-scored for the target: MCQ by chosen letter AND chosen option text (both
    readings must agree), numeric by final value. Anything else is unknown and
    enters the conservative/optimistic quality bounds.
    """
    if entry["duplicate_group_id"] == target["duplicate_group_id"] or entry["key_hash"] == target["key_hash"]:
        return REUSE_RECORDED
    family, gold, answer = target["task_family"], target.get("gold"), entry["routed_answer"]
    if family == "mcq" and gold:
        letter, status = choice_letter(answer)
        if status == "none":
            return REUSE_INCORRECT
        if status != "ok":
            return REUSE_UNKNOWN
        letter_ok = letter == gold
        entry_options = json.loads(entry.get("options_json") or "{}")
        target_options = json.loads(target.get("options_json") or "{}")
        text_ok = None
        if letter in entry_options and gold in target_options:
            text_ok = entry_options[letter] == target_options[gold]
        if letter_ok and text_ok is True:
            return REUSE_CORRECT
        if not letter_ok and text_ok is not True:
            return REUSE_INCORRECT
        return REUSE_UNKNOWN
    if family == "numeric" and gold:
        value = numeric_answer(answer)
        if value is None:
            return REUSE_INCORRECT
        expected, got = as_number(gold), as_number(value)
        if expected is not None and got is not None:
            return REUSE_CORRECT if math.isclose(expected, got, rel_tol=1e-9, abs_tol=1e-6) else REUSE_INCORRECT
        return REUSE_CORRECT if canonical_answer(value) == canonical_answer(gold) else REUSE_UNKNOWN
    return REUSE_UNKNOWN


# ------------------------------------------------------------------ policies --

@dataclass(frozen=True)
class Policy:
    name: str
    exact: bool
    semantic: bool
    candidate_limit: int = 5
    thresholds: dict = field(default_factory=lambda: {"easy": 0.80, "medium": 0.80, "hard": 0.80})
    rerank_mode: str = "off"
    rerank_similarity: float | None = None
    rerank_score: float | None = None
    capacity: int | None = None

    @property
    def caches(self):
        return self.exact or self.semantic

    def threshold(self, label):
        return float(self.thresholds.get(label, self.thresholds["medium"]))


def policy_from(name, spec, **overrides):
    spec = dict(spec, **overrides)
    mode = spec.get("rerank_mode", "off")
    if mode not in ("off", "shadow", "enforce"):
        raise ValueError(f"Unknown rerank mode {mode}")
    if mode != "off" and (spec.get("rerank_similarity") is None or spec.get("rerank_score") is None):
        raise ValueError("Reranker thresholds are required when the reranker is on")
    if spec.get("semantic") and not 1 <= int(spec.get("candidate_limit", 5)) <= 16:
        raise ValueError("candidate_limit must be within FastAPI rerank_max_candidates (16)")
    return Policy(name=name, exact=bool(spec["exact"]), semantic=bool(spec["semantic"]),
                  candidate_limit=int(spec.get("candidate_limit", 5)),
                  thresholds=dict(spec.get("thresholds", {"easy": 0.80, "medium": 0.80, "hard": 0.80})),
                  rerank_mode=mode, rerank_similarity=spec.get("rerank_similarity"),
                  rerank_score=spec.get("rerank_score"), capacity=spec.get("capacity"))


def config_policies(config, **overrides):
    return {name: policy_from(name, config["policies"][name], **overrides) for name in config["policies"]}


# --------------------------------------------------------------- the cache --

class MemoryCache:
    """In-memory cache_entry with the operational lookup semantics.

    exact(): newest updated row for the key (SELECT_EXACT_SQL ORDER BY updated_at).
    candidates(): exact cosine top-k, then per-row threshold, similarity desc
    (SELECT_SQL). HNSW is approximate; latency.py replays with real pgvector.
    Hits refresh updated_at (recordHit). Optional LRU eviction by updated_at.
    """

    def __init__(self, dim, capacity=None, initial=1024):
        self.dim, self.capacity = dim, capacity
        self.vectors = np.zeros((initial, dim), dtype=np.float32) if dim else None
        self.alive = np.zeros(initial, dtype=bool)
        self.rows, self.keys, self.labels, self.thresholds, self.updated = [], [], [], [], []
        self.by_key, self.clock, self.evictions = {}, 0, 0

    def __len__(self):
        return int(self.alive[: len(self.rows)].sum())

    def tick(self):
        self.clock += 1
        return self.clock

    def exact(self, key):
        ids = [i for i in self.by_key.get(key, ()) if self.alive[i]]
        if not ids:
            return None
        return max(ids, key=lambda i: (self.updated[i], i))

    def candidates(self, vector, k):
        n = len(self.rows)
        if not n or self.vectors is None:
            return []
        sims = self.vectors[:n] @ np.asarray(vector, dtype=np.float32)
        sims = np.where(self.alive[:n], sims, -np.inf)
        order = np.lexsort((np.arange(n), -sims))[: min(k, n)]
        top = [(int(i), float(sims[i])) for i in order if np.isfinite(sims[i])]
        passed = [(i, s) for i, s in top if s >= self.thresholds[i]]
        return sorted(passed, key=lambda t: (-t[1], t[0]))

    def touch(self, entry_id):
        self.updated[entry_id] = self.tick()

    def insert(self, row, key, vector, label, threshold):
        if self.capacity is not None and len(self) >= self.capacity:
            live = np.flatnonzero(self.alive[: len(self.rows)])
            victim = min(live, key=lambda i: (self.updated[i], i))
            self.alive[victim] = False
            self.evictions += 1
        i = len(self.rows)
        if i >= len(self.alive):
            grow = len(self.alive)
            self.alive = np.concatenate([self.alive, np.zeros(grow, dtype=bool)])
            if self.vectors is not None:
                self.vectors = np.concatenate([self.vectors, np.zeros((grow, self.dim), dtype=np.float32)])
        if self.vectors is not None:
            if vector is None:
                raise ValueError("A semantic cache insert needs the request embedding")
            self.vectors[i] = vector
        self.alive[i] = True
        self.rows.append(int(row))
        self.keys.append(key)
        self.labels.append(label)
        # cache_entry.threshold is REAL: 0.80 is stored as float32 and compared
        # against a double similarity, so keep the float32-rounded value.
        self.thresholds.append(float(np.float32(threshold)))
        self.updated.append(self.tick())
        self.by_key.setdefault(key, []).append(i)
        return i


# ------------------------------------------------------------------- traces --

def make_trace(pool_ids, length, repeat_fraction, seed):
    """Request order for one (repeat fraction, seed) condition.

    New questions follow one seed-specific permutation, so every repeat fraction
    of a seed draws its new questions as a prefix of the same order. Exactly
    round(r * length) positions (never the first) re-send a question already
    sent, chosen uniformly among the distinct questions seen so far.
    """
    pool_ids = np.asarray(pool_ids)
    repeats = int(round(repeat_fraction * length))
    unique = length - repeats
    if length < 1 or unique < 1 or unique > len(pool_ids):
        raise ValueError(f"Cannot build a trace of {length} with {unique} new questions from {len(pool_ids)}")
    order = np.random.default_rng([seed, 0]).permutation(pool_ids)[:unique]
    rng = np.random.default_rng([seed, 1, int(round(repeat_fraction * 1000))])
    repeat_positions = set((rng.choice(np.arange(1, length), size=repeats, replace=False) + 0).tolist()) if repeats else set()
    rows, seen, first, nxt = [], [], {}, 0
    for position in range(length):
        if position in repeat_positions:
            sample = seen[int(rng.integers(len(seen)))]
            rows.append((position, sample, True, first[sample]))
        else:
            sample = order[nxt]
            nxt += 1
            seen.append(sample)
            first[sample] = position
            rows.append((position, sample, False, position))
    trace = pd.DataFrame(rows, columns=["position", "sample_id", "is_repeat", "first_position"])
    trace.attrs.update(repeat_fraction=repeat_fraction, seed=seed, length=length)
    return trace


def trace_id(repeat_fraction, seed):
    return f"r{int(round(repeat_fraction * 100)):02d}-s{seed}"


# ------------------------------------------------------------------- replay --

class PairScorer:
    """Memoized cross-encoder calls keyed by the whole candidate list.

    The service scores all candidates of a request in one padded batch, and the
    padding can move scores in the last digits, so a pair score is only reused
    for the identical batch. `calls` counts service-equivalent calls made.
    """

    def __init__(self, score_fn, texts):
        self.score_fn, self.texts, self.memo, self.calls = score_fn, texts, {}, 0

    def __call__(self, query_row, candidate_rows):
        key = (int(query_row), tuple(int(c) for c in candidate_rows))
        if key not in self.memo:
            self.calls += 1
            scores = self.score_fn(self.texts[query_row], [self.texts[c] for c in candidate_rows])
            if len(scores) != len(candidate_rows) or not all(math.isfinite(s) for s in scores):
                raise ValueError("Reranker must return one finite score per candidate")
            self.memo[key] = [float(s) for s in scores]
        return self.memo[key]


def semantic_decision(policy, cache, row, vector, rerank, texts):
    """RerankGateService.evaluate on the CacheService candidate list."""
    candidates = cache.candidates(vector, policy.candidate_limit)
    info = {"candidate_count": len(candidates), "rerank_pairs": 0, "rerank_score": np.nan,
            "rerank_reason": "no_candidates", "cosine_best": candidates[0][1] if candidates else np.nan}
    if not candidates:
        return None, info
    cosine_best = candidates[0]
    if policy.rerank_mode == "off":
        info["rerank_reason"] = "disabled"
        return cosine_best, info
    scorable = [(e, s) for e, s in candidates if str(texts[cache.rows[e]]).strip()]
    if not scorable:
        info["rerank_reason"] = "no_scorable_candidate"
        return cosine_best, info
    scores = rerank(row, [cache.rows[e] for e, _ in scorable])
    info["rerank_pairs"] = len(scorable)
    accepted = -1
    for i, (_, similarity) in enumerate(scorable):
        if similarity >= policy.rerank_similarity and scores[i] >= policy.rerank_score and (accepted < 0 or scores[i] > scores[accepted]):
            accepted = i
    served = next((i for i, (e, _) in enumerate(scorable) if e == cosine_best[0]), -1)
    info["rerank_score"] = scores[served] if served >= 0 else max(scores)
    if policy.rerank_mode == "shadow":
        info["rerank_reason"] = "shadow"
        return cosine_best, info
    if accepted < 0:
        info["rerank_reason"] = "rejected"
        return None, info
    info["rerank_score"] = scores[accepted]
    info["rerank_reason"] = "accepted" if accepted == served else "accepted_reordered"
    return scorable[accepted], info


def replay(trace, pool, policy, vectors=None, rerank=None, cache=None):
    """Run one arm over one trace. Returns one log row per request."""
    index = {s: i for i, s in enumerate(pool.sample_id)}
    records = pool.to_dict("records")
    texts = pool.request_text.tolist()
    if policy.semantic and vectors is None:
        raise ValueError("Semantic arms need request embeddings")
    if policy.rerank_mode != "off" and rerank is None:
        raise ValueError("Reranked arms need a reranker")
    if cache is None and policy.caches:
        cache = MemoryCache(vectors.shape[1] if policy.semantic else 0, capacity=policy.capacity)
    log, outcome_memo = [], {}
    for request in trace.itertuples(index=False):
        row = index[request.sample_id]
        target = records[row]
        hit, kind, ops = None, "miss", {"exact_lookups": 0, "embeds": 0, "vector_searches": 0}
        info = {"candidate_count": 0, "rerank_pairs": 0, "rerank_score": np.nan, "rerank_reason": "not_evaluated",
                "cosine_best": np.nan}
        if policy.exact:
            ops["exact_lookups"] = 1
            entry = cache.exact(target["key_hash"])
            if entry is not None:
                hit, kind, similarity = entry, "exact", 1.0
        if hit is None and policy.semantic:
            ops["embeds"] = ops["vector_searches"] = 1
            chosen, info = semantic_decision(policy, cache, row, vectors[row], rerank, texts)
            if chosen is not None:
                hit, kind, similarity = chosen[0], "semantic", chosen[1]
        if hit is not None:
            cache.touch(hit)
            source = cache.rows[hit]
            key = (source, row)
            if key not in outcome_memo:
                outcome_memo[key] = reuse_outcome(records[source], target)
            reuse = outcome_memo[key]
            if reuse == REUSE_RECORDED:
                q_lo = q_hi = records[source]["routed_quality"]
                s_lo = s_hi = records[source]["routed_success"]
            else:
                q_lo = s_lo = 1.0 if reuse == REUSE_CORRECT else 0.0
                q_hi = s_hi = 0.0 if reuse == REUSE_INCORRECT else 1.0
            log.append({**_base(request, target, policy, kind, ops, info), "source_sample_id": records[source]["sample_id"],
                        "similarity": similarity, "relation_error": records[source]["duplicate_group_id"] != target["duplicate_group_id"],
                        "reuse": reuse, "llm_called": False, "routed": False, "llm_cost": 0.0, "llm_latency_s": 0.0,
                        "quality_lo": float(q_lo), "quality_hi": float(q_hi), "success_lo": float(s_lo), "success_hi": float(s_hi),
                        "inserted": False})
            continue
        inserted = False
        if policy.caches:
            vector = vectors[row] if policy.semantic else None
            cache.insert(row, target["key_hash"], vector, target["routed_label"], policy.threshold(target["routed_label"]))
            inserted = True
        log.append({**_base(request, target, policy, kind, ops, info), "source_sample_id": None, "similarity": np.nan,
                    "relation_error": False, "reuse": None, "llm_called": True, "routed": True,
                    "llm_cost": float(target["routed_actual_cost"]), "llm_latency_s": float(target["routed_latency_s"]),
                    "quality_lo": float(target["routed_quality"]), "quality_hi": float(target["routed_quality"]),
                    "success_lo": float(target["routed_success"]), "success_hi": float(target["routed_success"]),
                    "inserted": inserted})
    frame = pd.DataFrame(log)
    frame.attrs["evictions"] = getattr(cache, "evictions", 0) if cache is not None else 0
    return frame


def _base(request, target, policy, kind, ops, info):
    return {"arm": policy.name, "position": request.position, "sample_id": request.sample_id,
            "duplicate_group_id": target["duplicate_group_id"], "source_dataset": target["source_dataset"],
            "task_family": target["task_family"], "is_repeat": bool(request.is_repeat), "decision": kind,
            "routed_tier": target["routed_tier"], **ops, **info}


# ------------------------------------------------------------------ metrics --

SUM_COLUMNS = ("n", "hit", "hit_exact", "hit_semantic", "relation_error", "unknown", "incorrect", "llm_called",
               "llm_cost", "quality_lo", "quality_hi", "success_lo", "success_hi", "embeds", "vector_searches",
               "rerank_pairs", "exact_lookups")


def group_sums(log):
    frame = pd.DataFrame({
        "duplicate_group_id": log.duplicate_group_id, "n": 1, "hit": log.decision.ne("miss"),
        "hit_exact": log.decision.eq("exact"), "hit_semantic": log.decision.eq("semantic"),
        "relation_error": log.relation_error.astype(bool), "unknown": log.reuse.eq(REUSE_UNKNOWN),
        "incorrect": log.reuse.eq(REUSE_INCORRECT), "llm_called": log.llm_called.astype(bool),
        **{c: log[c] for c in ("llm_cost", "quality_lo", "quality_hi", "success_lo", "success_hi", "embeds",
                               "vector_searches", "rerank_pairs", "exact_lookups")}})
    return frame.groupby("duplicate_group_id", sort=True)[list(SUM_COLUMNS)].sum().astype(float)


def ratios(sums, baseline=None):
    """Metric dictionary from summed counts (works on bootstrap-weighted sums)."""
    s = {c: sums[c] for c in SUM_COLUMNS}
    div = lambda a, b: a / b if b else float("nan")
    out = {"requests": s["n"], "hit_rate": div(s["hit"], s["n"]), "exact_hit_rate": div(s["hit_exact"], s["n"]),
           "semantic_hit_rate": div(s["hit_semantic"], s["n"]),
           "false_reuse_among_hits": div(s["relation_error"], s["hit"]), "false_reuse_per_request": div(s["relation_error"], s["n"]),
           "unknown_among_hits": div(s["unknown"], s["hit"]), "incorrect_reuse_among_hits": div(s["incorrect"], s["hit"]),
           "llm_calls": s["llm_called"], "llm_cost": s["llm_cost"],
           "quality_conservative": div(s["quality_lo"], s["n"]), "quality_optimistic": div(s["quality_hi"], s["n"]),
           "success_conservative": div(s["success_lo"], s["n"]), "success_optimistic": div(s["success_hi"], s["n"]),
           "embeds": s["embeds"], "vector_searches": s["vector_searches"], "rerank_pairs": s["rerank_pairs"],
           "exact_lookups": s["exact_lookups"]}
    if baseline is not None:
        b = {c: baseline[c] for c in SUM_COLUMNS}
        out.update({"call_reduction": 1 - div(s["llm_called"], b["llm_called"]),
                    "llm_cost_reduction": 1 - div(s["llm_cost"], b["llm_cost"]),
                    "quality_change_conservative": div(s["quality_lo"], s["n"]) - div(b["quality_lo"], b["n"]),
                    "quality_change_optimistic": div(s["quality_hi"], s["n"]) - div(b["quality_hi"], b["n"])})
    return out


def paired_bootstrap(logs, baseline_arm, draws, seed, keys=("hit_rate", "false_reuse_among_hits", "call_reduction",
                                                             "llm_cost_reduction", "quality_change_conservative",
                                                             "quality_change_optimistic")):
    """Question-group paired bootstrap over arms that saw the same trace."""
    arms = list(logs)
    sums = {arm: group_sums(logs[arm]) for arm in arms}
    groups = sums[baseline_arm].index
    for arm in arms:
        if not sums[arm].index.equals(groups):
            raise ValueError("Arms must replay the same trace")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
    matrices = {arm: sums[arm].to_numpy() for arm in arms}
    point = {arm: ratios(sums[arm].sum(), sums[baseline_arm].sum()) for arm in arms}
    rows = []
    for arm in arms:
        values = {k: [] for k in keys}
        for w in weights:
            m = dict(zip(SUM_COLUMNS, w @ matrices[arm]))
            b = dict(zip(SUM_COLUMNS, w @ matrices[baseline_arm]))
            r = ratios(m, b)
            for k in keys:
                values[k].append(r[k])
        for k in keys:
            v = np.asarray(values[k], dtype=float)
            v = v[np.isfinite(v)]
            lo, hi = (np.quantile(v, [0.025, 0.975]) if len(v) else (np.nan, np.nan))
            rows.append({"arm": arm, "metric": k, "point": point[arm][k], "ci_low": float(lo), "ci_high": float(hi),
                         "finite_draws": int(len(v))})
    return pd.DataFrame(rows)


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def exact_arm_guard(log):
    """An exact hit must never reuse another question's answer (stop rule, every arm)."""
    bad = log[log.decision.eq("exact") & log.relation_error.astype(bool)]
    if len(bad):
        raise AssertionError(f"Exact cache reused another question {len(bad)} times; normalization bug")


# --------------------------------------------------------------------- D2 ----

D2_ROLES = ("paraphrase", "hard_negative")
YES, NO, UNSURE = "Y", "N", "U"


def select_d2_originals(pool, config, token_lengths):
    """Stratified, order-independent choice of D2 originals and their split.

    token_lengths maps sample_id -> BGE token count (special tokens included).
    The pool has already dropped questions over the service input limit; this
    re-checks it because a longer question can never be cached.
    """
    d2 = config["d2"]
    frame = pool.assign(tokens=pool.sample_id.map(token_lengths))
    if frame.tokens.isna().any():
        raise ValueError("Token length is required for every pool row")
    if (frame.tokens > config["service_input_tokens"]).any():
        raise ValueError("Pool still holds questions over the service input limit")
    frame["stratum"] = [_stratum(r) for r in frame.itertuples()]
    frame = frame[frame.routed_answer.fillna("").str.strip().ne("")]
    frame["_order"] = frame.sample_id.map(lambda s: digest([d2["selection_seed"], s]))
    chosen = []
    for stratum, quota in d2["strata"].items():
        part = frame[frame.stratum.eq(stratum)].sort_values("_order")
        if len(part) < quota:
            raise ValueError(f"D2 stratum {stratum} has {len(part)} eligible rows for quota {quota}")
        part = part.head(quota).copy()
        n_cal = int(round(quota * d2["calibration_fraction"]))
        part["d2_split"] = ["calibration"] * n_cal + ["test"] * (quota - n_cal)
        types = d2["negative_types"][stratum]
        part["negative_type"] = [types[i % len(types)] for i in range(len(part))]
        chosen.append(part)
    out = pd.concat(chosen, ignore_index=True).drop(columns="_order")
    out["group_id"] = [f"d2-{i:03d}" for i in range(len(out))]
    return out


def _stratum(row):
    if row.source_dataset == "xroutebench":
        return "xrb_" + ("mcq" if row.task_family == "mcq" else "numeric" if row.task_family == "numeric" else "open")
    if row.task_family in ("mcq", "mcq_no_gold"):
        return "rb_mcq"
    return "rb_math" if row.source_task == "grade-school-math" else "rb_open"


AUTHORING_COLUMNS = ["group_id", "d2_split", "stratum", "negative_type", "sample_id", "source_task", "original_text",
                     "cached_answer", "original_recorded_quality", "gold", "paraphrase_text", "hard_negative_text",
                     "author", "author_notes"]


def authoring_sheet(originals):
    return pd.DataFrame({
        "group_id": originals.group_id, "d2_split": originals.d2_split, "stratum": originals.stratum,
        "negative_type": originals.negative_type, "sample_id": originals.sample_id,
        "source_task": originals.source_task, "original_text": originals.request_text,
        "cached_answer": originals.routed_answer, "original_recorded_quality": originals.routed_quality,
        "gold": originals.gold, "paraphrase_text": "", "hard_negative_text": "", "author": "", "author_notes": ""})[AUTHORING_COLUMNS]


def validate_authoring(sheet, originals, token_count=None, limit=512):
    """Reject incomplete, trivially different or uncacheable variants.

    A variant over the service input limit would be rejected by the service, so
    it can never reach the cache and is not a valid D2 item.
    """
    problems = []
    filled = sheet[["group_id", "paraphrase_text", "hard_negative_text", "author"]]
    merged = originals[["group_id", "request_text", "stratum"]].merge(filled, on="group_id", how="left", validate="one_to_one")
    for r in merged.itertuples():
        for role, text in (("paraphrase", r.paraphrase_text), ("hard_negative", r.hard_negative_text)):
            if not isinstance(text, str) or not text.strip():
                problems.append((r.group_id, role, "empty"))
            elif question_hash(text) == question_hash(r.request_text):
                problems.append((r.group_id, role, "identical_after_whitespace_normalization"))
        if not isinstance(r.author, str) or not r.author.strip():
            problems.append((r.group_id, "author", "missing"))
        if token_count is not None:
            for role, text in (("paraphrase", r.paraphrase_text), ("hard_negative", r.hard_negative_text)):
                if isinstance(text, str) and text.strip() and token_count(text) > limit:
                    problems.append((r.group_id, role, f"exceeds_service_input_limit_{limit}"))
    if sheet.group_id.duplicated().any():
        problems.append(("*", "group_id", "duplicated"))
    return pd.DataFrame(problems, columns=["group_id", "field", "problem"])


QUALITY_FLAGS = ("low-quality trap", "source anomaly")


def quality_flags(notes):
    """Author-note tags that mark a group for the sensitivity analysis."""
    text = str(notes or "")
    return [flag for flag in QUALITY_FLAGS if flag in text]


def variants_from(sheet, originals):
    rows = []
    base = originals.set_index("group_id")
    notes = sheet.author_notes if "author_notes" in sheet else pd.Series("", index=sheet.index)
    for r, note in zip(sheet.itertuples(), notes):
        for role, text in (("paraphrase", r.paraphrase_text), ("hard_negative", r.hard_negative_text)):
            o = base.loc[r.group_id]
            rows.append({"item_id": f"{r.group_id}-{'p' if role == 'paraphrase' else 'n'}", "group_id": r.group_id,
                         "d2_split": o.d2_split, "stratum": o.stratum, "role": role,
                         "negative_type": o.negative_type if role == "hard_negative" else None,
                         "original_sample_id": o.sample_id, "variant_text": text,
                         "key_hash": question_hash(text), "author": r.author,
                         "quality_flags": ",".join(quality_flags(note))})
    return pd.DataFrame(rows)


ANNOTATION_COLUMNS = ["item_id", "original_text", "variant_text", "cached_answer", "original_gold",
                      "semantic_equivalent", "cached_answer_correct_for_target", "rationale", "annotator"]


def annotation_sheet(variants, originals, seed):
    """Blind sheet: the author's intended role is hidden and the order shuffled."""
    base = originals.set_index("group_id")
    frame = pd.DataFrame({
        "item_id": variants.item_id,
        "original_text": [base.loc[g].request_text for g in variants.group_id],
        "variant_text": variants.variant_text,
        "cached_answer": [base.loc[g].routed_answer for g in variants.group_id],
        "original_gold": [base.loc[g].gold for g in variants.group_id],
        "semantic_equivalent": "", "cached_answer_correct_for_target": "", "rationale": "", "annotator": ""})
    order = variants.item_id.map(lambda s: digest([seed, s])).sort_values().index
    return frame.loc[order, ANNOTATION_COLUMNS].reset_index(drop=True)


def cohen_kappa(a, b):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    labels = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == x).mean()) * float((b == x).mean()) for x in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def merge_annotations(first, second, adjudication=None):
    """Final labels: agreement, else the adjudicator's decision with a reason."""
    fields = ("semantic_equivalent", "cached_answer_correct_for_target")
    allowed = {"semantic_equivalent": {YES, NO}, "cached_answer_correct_for_target": {YES, NO, UNSURE}}
    a, b = first.set_index("item_id"), second.set_index("item_id")
    if set(a.index) != set(b.index):
        raise ValueError("Both annotators must label the same items")
    for sheet, name in ((a, "first"), (b, "second")):
        if sheet.annotator.fillna("").str.strip().eq("").any():
            raise ValueError(f"{name} sheet has rows without annotator")
        for f in fields:
            bad = ~sheet[f].isin(allowed[f])
            if bad.any():
                raise ValueError(f"{name} sheet has invalid {f} values: {sorted(set(sheet.loc[bad, f]))[:5]}")
    if set(a.annotator) & set(b.annotator):
        raise ValueError("The two sheets must come from different annotators")
    b = b.loc[a.index]
    out = pd.DataFrame(index=a.index)
    report = {}
    disagreements = []
    for f in fields:
        out[f"{f}_a"], out[f"{f}_b"] = a[f], b[f]
        report[f] = {"agreement": float((a[f] == b[f]).mean()), "cohen_kappa": cohen_kappa(a[f], b[f])}
        disagreements += [(i, f) for i in a.index[a[f] != b[f]]]
    adj = adjudication.set_index(["item_id", "field"]) if adjudication is not None and len(adjudication) else None
    for f in fields:
        final = a[f].where(a[f] == b[f])
        for i in a.index[a[f] != b[f]]:
            if adj is None or (i, f) not in adj.index:
                continue
            decision = adj.loc[(i, f)]
            if decision.final not in allowed[f] or not str(decision.reason).strip():
                raise ValueError(f"Adjudication for {i}/{f} needs a valid label and a reason")
            final.loc[i] = decision.final
        out[f] = final
    pending = pd.DataFrame(disagreements, columns=["item_id", "field"])
    if adj is not None:
        pending = pending[[k not in adj.index for k in zip(pending.item_id, pending.field)]]
    return out.reset_index(), pending, report


# ----------------------------------------------------------- D2 evaluation --

def candidate_table(query_vectors, cache_vectors, limit=16):
    """Top-`limit` cache rows per query by exact cosine (ties -> lower index)."""
    sims = query_vectors @ cache_vectors.T
    order = np.argsort(-sims, axis=1, kind="stable")[:, :limit]
    return order, np.take_along_axis(sims, order, axis=1)


def candidate_status(items, order, cache_frame, labels, originals):
    """Outcome if an item were served candidate j, precomputed once per item.

    Own original: relation error unless the item was judged semantically
    equivalent; answer status from the adjudicated reuse label. Any other cached
    question is a relation error; its answer is re-scored only for an
    equivalent paraphrase whose original has a gold answer, else unknown.
    """
    lab = labels.set_index("item_id").loc[items.item_id]
    equivalent = lab.semantic_equivalent.to_numpy() == YES
    correct = lab.cached_answer_correct_for_target.to_numpy()
    base = originals.set_index("sample_id")
    cache_records = cache_frame.to_dict("records")
    samples = cache_frame.sample_id.to_numpy()
    relation = np.ones(order.shape, dtype=bool)
    status = np.full(order.shape, REUSE_UNKNOWN, dtype=object)
    for i, item in enumerate(items.itertuples(index=False)):
        own = base.loc[item.original_sample_id].to_dict()
        target = dict(own, sample_id=item.item_id, duplicate_group_id=item.group_id, key_hash=item.key_hash)
        for j, row in enumerate(order[i]):
            if samples[row] == item.original_sample_id:
                relation[i, j] = not equivalent[i]
                status[i, j] = {YES: REUSE_CORRECT, NO: REUSE_INCORRECT}.get(correct[i], REUSE_UNKNOWN)
            elif item.role == "paraphrase" and equivalent[i]:
                status[i, j] = reuse_outcome(cache_records[row], target)
    return relation, status


def d2_decide(sims, scores, kind, k, cosine, rerank_similarity=None, rerank_score=None):
    """Vectorised operational decision: chosen candidate position, -1 = miss.

    'cosine': the most similar candidate if it clears the threshold.
    'rerank': among the top-k with sim >= cosine, the highest reranker score
    that also has sim >= rerank_similarity and score >= rerank_score.
    """
    s = sims[:, :k]
    eligible = s >= cosine
    if kind == "cosine":
        return np.where(eligible[:, 0], 0, -1)
    sc = scores[:, :k]
    ok = eligible & (s >= rerank_similarity) & (sc >= rerank_score)
    best = np.argmax(np.where(ok, sc, -np.inf), axis=1)
    return np.where(ok.any(axis=1), best, -1)


def d2_outcomes(items, labels, position, order, sims, relation, status, cache_frame):
    out = items[["item_id", "group_id", "role", "stratum", "negative_type", "original_sample_id", "d2_split"]].copy()
    rows = np.arange(len(items))
    j = np.maximum(position, 0)
    hit = position >= 0
    out["hit"] = hit
    out["similarity"] = np.where(hit, sims[rows, j], np.nan)
    out["hit_sample_id"] = np.where(hit, cache_frame.sample_id.to_numpy()[order[rows, j]], None)
    out["hit_own_original"] = hit & (out.hit_sample_id.to_numpy() == out.original_sample_id.to_numpy())
    out["relation_error"] = hit & relation[rows, j]
    out["answer_status"] = np.where(hit, status[rows, j], None)
    lab = labels.set_index("item_id").loc[items.item_id]
    out["semantic_equivalent"] = lab.semantic_equivalent.to_numpy() == YES
    out["reusable"] = lab.cached_answer_correct_for_target.to_numpy() == YES
    return out


def d2_metrics(outcomes):
    hits = outcomes[outcomes.hit]
    f, h = int(hits.relation_error.sum()), int(len(hits))
    lo, hi = clopper_pearson(f, h)
    reusable = outcomes[outcomes.reusable]
    wrong = int(hits.answer_status.eq(REUSE_INCORRECT).sum())
    unknown = int(hits.answer_status.eq(REUSE_UNKNOWN).sum())
    negatives, paraphrases = outcomes[outcomes.role.eq("hard_negative")], outcomes[outcomes.role.eq("paraphrase")]
    return {"items": int(len(outcomes)), "hits": h, "false_reuse": f,
            "false_reuse_among_hits": f / h if h else float("nan"), "false_reuse_ci95_low": lo, "false_reuse_ci95_high": hi,
            "reusable_items": int(len(reusable)), "reusable_hits": int(reusable.hit_own_original.sum()),
            "reusable_recall": float(reusable.hit_own_original.mean()) if len(reusable) else float("nan"),
            "wrong_answers_among_hits": wrong, "unknown_answers_among_hits": unknown,
            "answer_error_among_hits_conservative": (wrong + unknown) / h if h else float("nan"),
            "answer_error_among_hits_optimistic": wrong / h if h else float("nan"),
            "hard_negative_hit_rate": float(negatives.hit.mean()) if len(negatives) else float("nan"),
            "paraphrase_hit_rate": float(paraphrases.hit.mean()) if len(paraphrases) else float("nan")}


def calibration_grid(config):
    c = config["calibration"]
    cos = np.round(np.arange(c["cosine_start"], c["cosine_stop"] + 1e-9, c["cosine_step"]), 6)
    grid = [{"kind": "cosine", "k": 1, "cosine": float(x)} for x in cos]
    for k in c["candidate_limits"]:
        for x in cos:
            for y in c["rerank_scores"]:
                grid.append({"kind": "rerank", "k": int(k), "cosine": float(x), "rerank_similarity": float(x),
                             "rerank_score": float(y)})
    return grid


def evaluate_spec(spec, items, labels, order, sims, scores, relation, status, cache_frame):
    position = d2_decide(sims, scores, spec["kind"], spec["k"], spec["cosine"],
                         spec.get("rerank_similarity"), spec.get("rerank_score"))
    return d2_outcomes(items, labels, position, order, sims, relation, status, cache_frame)


def calibrate(items, labels, order, sims, scores, relation, status, cache_frame, config):
    """Pick the policy with most reusable hits subject to F/H <= limit.

    Ties prefer no reranker, then fewer candidates, then stricter thresholds.
    Returns (chosen spec or None, full grid table).
    """
    limit = config["calibration"]["max_false_reuse_among_hits"]
    rows = []
    for spec in calibration_grid(config):
        m = d2_metrics(evaluate_spec(spec, items, labels, order, sims, scores, relation, status, cache_frame))
        rows.append({**spec, **m})
    table = pd.DataFrame(rows)
    feasible = table[(table.hits > 0) & (table.false_reuse_among_hits <= limit)].copy()
    if feasible.empty:
        return None, table
    feasible["_rr"] = feasible.kind.eq("rerank").astype(int)
    feasible["_score"] = feasible.rerank_score.fillna(0.0) if "rerank_score" in feasible else 0.0
    best = feasible.sort_values(["reusable_hits", "_rr", "k", "cosine", "_score"],
                                ascending=[False, True, True, False, False]).iloc[0]
    spec = {"kind": str(best.kind), "k": int(best.k), "cosine": float(best.cosine)}
    if best.kind == "rerank":
        spec.update(rerank_similarity=float(best.rerank_similarity), rerank_score=float(best.rerank_score))
    return spec, table


def production_specs(config):
    """D2 decision specs equivalent to the operational C2/C3 policies."""
    c2, c3 = config["policies"]["C2"], config["policies"]["C3"]
    floor2, floor3 = min(c2["thresholds"].values()), min(c3["thresholds"].values())
    if len(set(c2["thresholds"].values())) != 1 or len(set(c3["thresholds"].values())) != 1:
        raise ValueError("Per-tier thresholds differ; D2 production specs assume one value")
    return {"C2": {"kind": "cosine", "k": int(c2["candidate_limit"]), "cosine": floor2},
            "C3": {"kind": "rerank", "k": int(c3["candidate_limit"]), "cosine": floor3,
                   "rerank_similarity": c3["rerank_similarity"], "rerank_score": c3["rerank_score"]}}


def calibrated_policy_spec(spec):
    """Operational policy dict for D1 from a frozen calibration choice."""
    if spec is None:
        return {"exact": True, "semantic": False}
    thresholds = {label: spec["cosine"] for label in LABELS}
    if spec["kind"] == "cosine":
        return {"exact": True, "semantic": True, "candidate_limit": 5, "thresholds": thresholds, "rerank_mode": "off"}
    return {"exact": True, "semantic": True, "candidate_limit": spec["k"], "thresholds": thresholds,
            "rerank_mode": "enforce", "rerank_similarity": spec["rerank_similarity"], "rerank_score": spec["rerank_score"]}
