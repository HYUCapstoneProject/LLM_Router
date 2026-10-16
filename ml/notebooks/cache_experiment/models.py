"""BGE embedder and cross-encoder reranker exactly as LLM_Router-serving runs them.

Copies the arithmetic of TransformersBgeEmbedder.embed (one message per call,
truncation at 512, CLS pooling, L2 normalisation) and
TransformersCrossEncoderReranker.rerank (pair batch, longest-first truncation at
512, sigmoid of the single logit). Batching changes padding and can move scores
in the last digits, so the default is one call per message like the service.
check_production_parity() compares against scores the deployed stack returned;
it is the gate for running on a GPU (Colab) instead of the service's CPU.
"""
from __future__ import annotations

import math
import platform
import time

import numpy as np


def runtime_versions(config, allow_mismatch=False):
    import torch
    import transformers
    found = {"torch": torch.__version__.split("+")[0], "transformers": transformers.__version__}
    expected = {k: config["runtime_pins"][k] for k in ("torch", "transformers")}
    mismatch = {k: (found[k], expected[k]) for k in expected if found[k] != expected[k]}
    if mismatch and not allow_mismatch:
        raise RuntimeError(f"Runtime differs from the serving image {mismatch}; install requirements.txt")
    return {**found, "expected": expected, "mismatch": mismatch, "python": platform.python_version(),
            "platform": platform.platform()}


def _device(torch, name):
    # Full fp32 on GPU too: TF32 matmuls would move scores far more than the
    # parity tolerance allows.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Embedder:
    def __init__(self, config, device="auto", local_files_only=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        spec = config["models"]["embed"]
        kwargs = {"revision": spec["revision"], "local_files_only": local_files_only}
        self.torch, self.spec = torch, spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec["id"], **kwargs)
        self.model = AutoModel.from_pretrained(spec["id"], use_safetensors=True, **kwargs)
        self.device = _device(torch, device)
        self.model.to(self.device).eval()
        if int(self.model.config.hidden_size) != spec["dimension"]:
            raise RuntimeError("BGE hidden size differs from the API contract")

    def token_count(self, text):
        return len(self.tokenizer(text, truncation=False)["input_ids"])

    def embed(self, text):
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.spec["max_length"])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            cls = self.model(**inputs).last_hidden_state[:, 0]
            vector = torch.nn.functional.normalize(cls, p=2, dim=1)[0].detach().to("cpu", dtype=torch.float32).numpy()
        if vector.shape != (self.spec["dimension"],) or not np.isfinite(vector).all():
            raise RuntimeError("BGE returned an invalid embedding")
        if not math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise RuntimeError("BGE returned a non-normalised vector")
        return vector

    def embed_all(self, texts, progress=None):
        out = np.zeros((len(texts), self.spec["dimension"]), dtype=np.float32)
        started = time.perf_counter()
        for i, text in enumerate(texts):
            out[i] = self.embed(text)
            if progress and (i + 1) % progress == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"embedded {i + 1}/{len(texts)} ({rate:.1f}/s)")
        return out


class Reranker:
    def __init__(self, config, device="auto", local_files_only=False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        spec = config["models"]["rerank"]
        kwargs = {"revision": spec["revision"], "local_files_only": local_files_only}
        self.torch, self.spec = torch, spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec["id"], **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(spec["id"], **kwargs)
        if int(self.model.config.num_labels) != 1:
            raise RuntimeError("Reranker checkpoint must expose exactly one logit")
        self.device = _device(torch, device)
        self.model.to(self.device).eval()

    def score(self, query, candidates):
        torch = self.torch
        inputs = self.tokenizer([query] * len(candidates), list(candidates), padding=True, truncation=True,
                                max_length=self.spec["max_length"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            scores = torch.sigmoid(self.model(**inputs).logits[:, 0]).tolist()
        if len(scores) != len(candidates) or not all(math.isfinite(s) for s in scores):
            raise RuntimeError("Reranker returned invalid scores")
        return [float(s) for s in scores]

    def score_matrix(self, queries, candidate_lists, progress=None):
        """One service-shaped call per query; rows padded with NaN."""
        width = max((len(c) for c in candidate_lists), default=0)
        out = np.full((len(queries), width), np.nan)
        for i, (query, candidates) in enumerate(zip(queries, candidate_lists)):
            out[i, : len(candidates)] = self.score(query, candidates)
            if progress and (i + 1) % progress == 0:
                print(f"reranked {i + 1}/{len(queries)}")
        return out


class Router:
    """The operational ModernBERT classifier, as TransformersModernBertClassifier runs it.

    package_dir defaults to config router.package_path under data_root. Identity
    is the SHA of the four model files, so the byte-identical R0 copy on Drive
    can stand in for the approved package on Colab.
    """

    LABELS = ("easy", "medium", "hard")

    def __init__(self, config, data_root=None, device="auto", package_dir=None):
        import torch
        from pathlib import Path
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import cache_experiment as ce
        spec = config["router"]
        path = Path(package_dir) if package_dir else Path(data_root) / spec["package_path"]
        for name, expected in spec["file_sha256"].items():
            ce.file_pin(path / name, expected)
        if (path / "model_manifest.json").is_file() and ce.read(path / "model_manifest.json")["model_version"] != spec["model_version"]:
            raise RuntimeError("Router package is not the operational model version")
        self.path, self.torch, self.max_length = str(path), torch, spec["logging_max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.device = _device(torch, device)
        self.model.to(self.device).eval()

    def classify(self, text):
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        return self.LABELS[int(torch.argmax(probabilities).item())]


def check_production_parity(questions, probes, embedder, reranker, config):
    """Recompute what the deployed stack returned on 2026-08-30.

    Replays experiments/validate_reranker_endtoend.py: the 19 `base` questions
    are the cache, each probe takes the top-5 seeds, the ones with cosine >= 0.80
    go to the reranker in one batch (same padding as the service call), and the
    stored score is the accepted best or, if none is accepted, the maximum.
    """
    import pandas as pd
    t = probes["thresholds"]
    tol_cos, tol_rr = config["parity"]["cosine_abs_tolerance"], config["parity"]["rerank_abs_tolerance"]
    seeds = [q["text"] for q in questions["questions"] if q["kind"] == "base"]
    if len(seeds) != probes["seeds"]:
        raise ValueError("questions-v2.json base count differs from the recorded run")
    seed_vectors = np.stack([embedder.embed(s) for s in seeds])
    rows = []
    for probe in probes["rows"]:
        sims = seed_vectors @ embedder.embed(probe["text"])
        order = np.argsort(-sims)[:5]
        passed = [int(j) for j in order if sims[j] >= t["cosine"]]
        best_score, accepted = None, None
        if passed:
            scores = reranker.score(probe["text"], [seeds[j] for j in passed])
            for j, score in zip(passed, scores):
                if sims[j] >= t["rerank_cosine"] and score >= t["rerank_score"] and (best_score is None or score > best_score):
                    best_score, accepted = score, j
            if best_score is None:
                best_score = max(scores)
        row = {"id": probe["id"], "kind": probe["kind"], "production_top": probe["top_candidate"],
               "local_top": seeds[int(order[0])], "production_cosine": probe["top_similarity"],
               "local_cosine": float(sims[order[0]]), "production_rerank": probe["rerank_score"],
               "local_rerank": best_score, "production_hit": probe["rerank_hit"], "local_hit": accepted is not None,
               "production_cosine_hit": probe["cosine_hit"], "local_cosine_hit": bool(passed)}
        row["cosine_diff"] = abs(row["local_cosine"] - row["production_cosine"])
        both = row["production_rerank"] is not None and row["local_rerank"] is not None
        row["rerank_diff"] = abs(row["local_rerank"] - row["production_rerank"]) if both else (
            0.0 if row["production_rerank"] is None and row["local_rerank"] is None else float("inf"))
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {
        "probes": int(len(frame)),
        "top_candidate_mismatches": int((frame.production_top != frame.local_top).sum()),
        "max_cosine_diff": float(frame.cosine_diff.max()),
        "max_rerank_diff": float(frame.rerank_diff.max()),
        "cosine_decision_mismatches": int((frame.production_cosine_hit != frame.local_cosine_hit).sum()),
        "rerank_decision_mismatches": int((frame.production_hit != frame.local_hit).sum()),
    }
    summary["passed"] = bool(summary["top_candidate_mismatches"] == 0 and summary["max_cosine_diff"] <= tol_cos
                             and summary["max_rerank_diff"] <= tol_rr and summary["cosine_decision_mismatches"] == 0
                             and summary["rerank_decision_mismatches"] == 0)
    return summary, frame


def threshold_margin_report(values, thresholds):
    """How many values sit within 1e-4 of a threshold (numeric-noise exposure)."""
    values = np.asarray(values, dtype=float)
    return {str(t): int(np.sum(np.abs(values - t) < 1e-4)) for t in thresholds}

