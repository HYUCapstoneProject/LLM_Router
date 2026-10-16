"""EV-10 04_common_track: paper routers and R1-BGE on the shared BGE embeddings.

Every method learns from the same 03_ours training rows (blended: four sources; kNN-A: the A split
only) and treats the three model-group tiers as three virtual models whose performance is the group
success probability. External code is not vendored: `fetch` downloads the needed files at pinned
commits and checks each file's SHA-256, and `run` imports those original classes, replacing only
their API, remote-embedding and file plumbing with the precomputed 01_embed matrix. Deviations are
listed in DEVIATIONS and copied into every output.

`run` writes raw per-tier scores for every dev, test and RouterArena question, once per setting
(at most eight per method), to <output>/<method>/<setting>/. 07_lock calibrates them per source on
dev and picks one setting per method. Test outcomes are never read here.

`build-outcomes` (local, CPU, before `run`) writes the per-model success of the training rows that
GraphRouter's model nodes need, to <run>/common-inputs/. Only training rows are read.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import pickle
import random
import sys
import time
import types
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

TIERS = ("low", "medium", "high")
SOURCES = ("routerbench", "xroutebench", "llmrouterbench", "sprout")
SOURCE_NAMES = {"routerbench": "RouterBench", "xroutebench": "xRouteBench",
                "llmrouterbench": "LLMRouterBench", "sprout": "SPROUT"}
SCORED_ROLES = ("dev", "test", "diagnostic")
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
SEED = 42

EXTERNAL = {
    "routerbench": {
        "repo": "withmartian/routerbench", "commit": "cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c", "license": "MIT",
        "files": {
            "LICENSE": "5a73d9410eb4f288d95245e5ec15ba55b2a5a080226d98869b52ba72379938bf",
            "routers/abstract_router.py": "f8f724cac79a09183c223d64e4115cb6d62ed3c0e38d42670a2b445a541eacd0",
            "routers/common.py": "27f853b177840036dbe2876c2fd0ac8fa68389f5d294fb73d94e30dd7c7105ec",
            "routers/knn_router.py": "998047b511e0e83b087c0ceace0fcd339e48c1fcda8a749bb60de8bc97fc131c",
            "routers/mlp_router.py": "0272729f3da51051f93f13b87296dc8cf189b0037a731ec15c8a2b399a408178",
            "routers/svm_router.py": "fa89a00614d6651c0d1c95238b31624d0eb7f86eb7fab836f986d67510b5e0d6",
        },
    },
    "llmrouter": {
        "repo": "ulab-uiuc/LLMRouter", "commit": "c65ee0122bc9e84f71d571fcdec4a948481b6e88", "license": "MIT",
        "files": {
            "LICENSE": "905c5b3e88f78dc96ed4d0b8ec1c0a20fbb3122988b7d992be64d751836ecc92",
            "llmrouter/models/base_trainer.py": "8bcde8ea1847646e9b7d60b03321b94d5b05363c1f0c52f63c38a771ea1c8af4",
            "llmrouter/models/mfrouter/__init__.py": "4e78bef0c5bbef7c61f8e00acfb4245cb123c33b147f9c16197422b23d0c34a6",
            "llmrouter/models/mfrouter/router.py": "866c0bd6d4427ebd049fcac9d47dec649077aeaafffd221193e33d466259f58e",
            "llmrouter/models/mfrouter/trainer.py": "7fcd4ae2c187a7c3a13972f949c1087066faabdfe0e386c3cb7e3a5b6e4889c9",
            "llmrouter/models/graphrouter/__init__.py": "068e349283706cff39dc886a74a4ddb48669a3b08dbcd43ef695c5da61b3a79a",
            "llmrouter/models/graphrouter/router.py": "7bef72791e0f772b6a88882ce14f9be4752625dc392e8fbfb42616af244e5f1b",
            "llmrouter/models/graphrouter/trainer.py": "5667a5ce86176e20afdb35987c8b2f708c6e1429a4a721ee4ea1253ca702114d",
            "llmrouter/models/graphrouter/graph_nn.py": "7581b2633e7ca18604508e61593aa9c2ec6d43375f8826a7cb199ba84eecee40",
            "llmrouter/models/hybrid_llm/__init__.py": "1953758336fbcd264432542ef98b0872a56e1b3684ccb7c00ca644acdddaf4bf",
            "llmrouter/models/hybrid_llm/router.py": "fd1a7bdae629393fabaf8edfe48b6c63ab5f8e312ad83dfe6b22e2b7c37f446f",
            "llmrouter/models/hybrid_llm/trainer.py": "34d702ea7ab265068bfba5797cca50a7c5b6e4dc69977256d174e0c992f48fa8",
            "llmrouter/models/largest_llm/__init__.py": "f43543906f9113fd5f3f5d9063ba868c2133f975b1af06e5a50f7816f2571a4a",
            "llmrouter/models/largest_llm/router.py": "07c87d9d76dcf3851cb13f6e2a9ca99921bfe3e9506a7734e7a69b9852f41c35",
        },
    },
    "graphrouter": {
        "repo": "ulab-uiuc/GraphRouter", "commit": "0edfb8758d45acd93c8029354867ffb49bb8a97f", "license": "MIT",
        "files": {
            "LICENSE": "905c5b3e88f78dc96ed4d0b8ec1c0a20fbb3122988b7d992be64d751836ecc92",
            "model/graph_nn.py": "573baddfe7b89f94014bcb80c335a5091a732389a32c43143c0506e4f56ba9db",
        },
    },
    "irt_router": {
        "repo": "Mercidaiha/IRT-Router", "commit": "e8f258ced4ec3c40d795403603acd8c1cdfb994d",
        "license": "none (research use only, not for the service)",
        "files": {"router/MIRT.py": "58883f95f5654e941663c5b651e17c172fd198e94bae4e7c252f73b136bf4758"},
    },
    "avengers_pro": {
        "repo": "ZhangYiqun018/AvengersPro", "commit": "dfb60b156c4b6011b89edfc8516aa16d8e4f4d38",
        "license": "none (research use only, not for the service)",
        "files": {
            "config.py": "970a36ce8132ca59bf39d5ff7a90900d340989fa8d4296071ce4cb6605ae2044",
            "simple_cluster_router.py": "efeeb6f8de18fc0415cf21ddd55bdc01e309b016820b6e6bfaacc9e37cae41e4",
            "balance_cluster_router.py": "2ce06c01b9ffbbc4cbe5c89f53441a4bf7bf69de20f760f24bc9f2d0d26ec6df",
        },
    },
}

# Settings per method (plan 4.3: at most eight). Values come from each repository's own defaults
# and example configs; the source of each list is noted.
SETTINGS = {
    # routerbench configs/evaluate_routers.yaml knn.neighbors [5, 10, 40] + KNNRouter default 50
    "knn": [{"n_neighbors": k} for k in (5, 10, 40, 50)],
    "knn_a": [{"n_neighbors": k} for k in (5, 10, 40, 50)],
    # evaluate_routers.yaml mlp.hidden_layer_sizes [100, [100, 100]] + MLPRouter default (100, 100, 100)
    "mlp": [{"hidden_layer_sizes": h} for h in ((100,), (100, 100), (100, 100, 100))],
    "svm": [{}],  # SVMRouter has no options: LinearSVR(random_state=1234)
    # LLMRouter configs/model_config_train/mfrouter.yaml (epochs 5) + MFRouterTrainer default (3)
    "mf": [{"epochs": 5}, {"epochs": 3}],
    # impl=llmrouter: LLMRouter graphrouter.yaml with three tier nodes. impl=official: the GraphRouter repository
    # (configs/config.yaml: hidden 8, lr 3e-4, 1000 epochs, batch 32, mask .5) with one node per model and tiers
    # formed at the end; its one-hot "best model" label, or per-edge success (partial borrowing; 100 = the
    # epoch count checked in the feasibility run). See DEVIATIONS.
    "graphrouter": [
        {"impl": "llmrouter"},
        {"impl": "official", "label": "best", "epochs": 1000, "checkpoint": "best_f1"},
        {"impl": "official", "label": "success", "epochs": 1000, "checkpoint": "last"},
        {"impl": "official", "label": "success", "epochs": 100, "checkpoint": "last"},
    ],
    # hybrid_llm.yaml uses probabilistic (tau .1); the other two modes are the router's alternatives
    "hybrid_llm": [{"router_mode": m} for m in ("probabilistic", "deterministic", "transformed")],
    # test_router.py --lamda default .1 (cold-start warm-up); 0 = off. One training serves all three.
    "irt_router": [{"lamda": v} for v in (0.0, 0.1, 0.3)],
    # balance_config.json (25 clusters, top_k 3, beta 9, alpha .45); README (alpha .7, example 64 clusters);
    # alpha 1 = performance only (SimpleClusterRouter ranking)
    "avengers_pro": [{"n_clusters": k, "performance_weight": a} for k, a in ((25, 0.45), (25, 0.7), (25, 1.0), (64, 0.7))],
    # ours: same target and source weighting as R1-group, fixed BGE input
    "r1_bge": [{"hidden": h, "epochs": e, "source_weight": w}
               for h in (256, 1024) for e in (5, 20) for w in (True, False)],
}

DEVIATIONS = {
    "all": [
        "Query input is the 01_embed BGE-large-en-v1.5 vector (512 tokens); no API or remote embedding calls.",
        "Each method outputs raw per-tier scores; the routing decision is the shared rule of 07_lock/08_test.",
        "No method sees a task name at inference.",
    ],
    "knn": ["RouterBench KNNRouter with the embedding cache replaced by the 01_embed matrix; calc_performance_scores is the score."],
    "knn_a": ["Same as knn, trained on the A split only (RouterBench + xRouteBench)."],
    "mlp": ["RouterBench MLPRouter with the embedding cache replaced; get_model_to_route_to is the score."],
    "svm": ["RouterBench SVMRouter with the embedding cache replaced; get_model_to_route_to is the score."],
    "mf": ["LLMRouter MFRouter + MFRouterTrainer; text_dim 1024 (BGE instead of Longformer); score = score_all per question."],
    "graphrouter": [
        "impl=llmrouter: LLMRouter GraphRouter + GraphTrainer as configured, with the three tiers as nodes (BGE vectors of "
        "the group descriptions). New questions are scaled with the MinMaxScaler fitted on the training questions (the "
        "original refits it on [scaled training rows; raw new rows]); score = sigmoid edge score (predict returns only the "
        "argmax). On the full training set every tier node sums ~25k question messages, predictions saturate after the "
        "first update (BCE ~33) and all scores are 0: reproduction failure.",
        "impl=official: the GraphRouter repository's EncoderDecoderNet / GNN_prediction unchanged. As in the paper there is "
        "one node per model (kept frontier models of the source, BGE vector of 'name, tier, relative price') and one graph "
        "per model pool; edges carry [performance, relative cost, performance] (Performance First). Task features are "
        "zero (no task names). Training uses training questions only, 1/8 held out as the official validation split; new "
        "questions are added as nodes without visible edges. Tiers are formed only at the end: tier score = mean "
        "predicted edge score of the tier's models. RouterArena questions belong to no pool and are not scored.",
        "label=best,checkpoint=best_f1 is the paper's label (one-hot best model, ties to the cheaper model) and its "
        "macro-F1 checkpoint. With 0/1 outcomes the best model is nearly always the cheapest, so this label carries "
        "little per-question information (feasibility run: dev Spearman with the tier probability about 0).",
        "label=success,checkpoint=last is partial borrowing: the same network trained on each question-model success "
        "(BCE), final-epoch weights.",
        "A missing model record (LLMRouterBench, 2% of cells) is filled with that row's group success probability "
        "from 03_ours, so a tier's mean over its models equals the R1-group target.",
    ],
    "hybrid_llm": [
        "LLMRouter HybridLLMRouter + HybridLLMTrainer; small model = low group, large model = high group (size tags 1B/3B).",
        "MLPRegressor random_state=42 added (the YAML leaves it unseeded). Single score = P(small is enough).",
    ],
    "irt_router": [
        "IRT-Router MIRT (latent 25, 9 epochs, batch 512, lr .001, train_mirt.py); LLM vectors = BGE of the group descriptions.",
        "Cold-start warm-up as in utils/cold_test.py (mean of 5 nearest training questions) for every scored question.",
        "Also stores the item difficulty b for H3.",
    ],
    "avengers_pro": [
        "Avengers-Pro BalanceClusterRouter clustering and balance ranking; cost = service cost .2/.6/1.0.",
        "Routing score re-implemented from route_queries_batch (same formula) to return every tier's score, not only the top one.",
    ],
    "r1_bge": ["Ours: MLP on the fixed BGE vector, soft BCE on the three group success probabilities (R1-group target)."],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setting_id(method: str, setting: dict) -> str:
    if not setting:
        return "default"
    parts = []
    for key, value in setting.items():
        if isinstance(value, (tuple, list)):
            value = "-".join(str(v) for v in value)
        elif isinstance(value, bool):
            value = "on" if value else "off"
        parts.append(f"{key}={value}")
    return ",".join(parts)


# ---------------------------------------------------------------------------------------------
# External code
# ---------------------------------------------------------------------------------------------

def fetch(dest: Path, external: dict | None = None) -> dict:
    """Download the pinned files and verify every SHA-256 before anything imports them."""
    external = EXTERNAL if external is None else external
    for key, spec in external.items():
        for path, digest in spec["files"].items():
            target = dest / key / path
            if not target.exists():
                url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/{path}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise SystemExit(f"pinned file changed: {spec['repo']}@{spec['commit']}:{path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    return verify_external(dest, external)


def verify_external(dest: Path, external: dict | None = None) -> dict:
    external = EXTERNAL if external is None else external
    for key, spec in external.items():
        for path, digest in spec["files"].items():
            if sha256_file(dest / key / path) != digest:
                raise SystemExit(f"external file does not match its pin: {key}/{path}")
    return {key: {k: spec[k] for k in ("repo", "commit", "license")} | {"files": spec["files"]}
            for key, spec in external.items()}


def _unused(*args, **kwargs):
    raise RuntimeError("API, remote-embedding and Longformer paths of the external routers are not used in EV-10")


def _module(name: str, path: Path | None = None, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    if path is not None:
        module.__path__ = [str(path)]
    return module


def _load_file(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def temporary_modules(stubs: dict):
    """Install stub modules for an import and restore whatever was there afterwards."""
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class ArrayCache:
    """Stands in for RouterBench's EmbeddingCache: prompts are sample ids looked up in the 01_embed matrix."""

    def __init__(self, ids, vectors: np.ndarray):
        self.index = {sample_id: i for i, sample_id in enumerate(ids)}
        self.vectors = vectors

    def batch_get_embedding(self, prompts: tuple, embedding_model: str) -> np.ndarray:
        return self.vectors[[self.index[p] for p in prompts]]


def load_routerbench(dest: Path) -> types.SimpleNamespace:
    root = dest / "routerbench"
    stubs = {
        "embedding": _module("embedding"),
        "embedding.cache": _module("embedding.cache", EmbeddingCache=_unused),
        # only feeds average response length, which matters only for willingness_to_pay (unused)
        "tokencost": _module("tokencost", count_string_tokens=lambda text, model: len(str(text).split())),
        "routers": _module("routers", root / "routers"),
    }
    with temporary_modules(stubs):
        for name in ("routers.common", "routers.abstract_router", "routers.knn_router",
                     "routers.mlp_router", "routers.svm_router"):
            sys.modules.pop(name, None)
        knn = importlib.import_module("routers.knn_router")
        mlp = importlib.import_module("routers.mlp_router")
        svm = importlib.import_module("routers.svm_router")
        for name in ("routers.common", "routers.abstract_router", "routers.knn_router",
                     "routers.mlp_router", "routers.svm_router"):
            sys.modules.pop(name, None)
    return types.SimpleNamespace(KNNRouter=knn.KNNRouter, MLPRouter=mlp.MLPRouter, SVMRouter=svm.SVMRouter)


def _save_model(obj, path):
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".pkl"):
        with open(path, "wb") as handle:
            pickle.dump(obj, handle)
    else:
        torch.save(obj, path)


def _load_model(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def register_llmrouter(dest: Path):
    """Make `llmrouter.*` importable from the pinned files with stubs for its YAML loader, API and Longformer
    helpers; returns the stub MetaRouter class whose `context` supplies what the original loader would set."""
    import torch

    class MetaRouter(torch.nn.Module):
        """Replaces LLMRouter's MetaRouter, which reads a YAML file and loads data from disk.
        Attributes the original loader would set come from MetaRouter.context."""
        context: dict = {}

        def __init__(self, model, yaml_path=None, resources=None):
            super().__init__()
            self.model = model
            for key, value in MetaRouter.context.items():
                setattr(self, key, value)

    root = dest / "llmrouter" / "llmrouter"
    for name in [n for n in sys.modules if n == "llmrouter" or n.startswith("llmrouter.")]:
        sys.modules.pop(name)
    sys.modules["llmrouter"] = _module("llmrouter", root)
    sys.modules["llmrouter.models"] = _module("llmrouter.models", root / "models")
    sys.modules["llmrouter.utils"] = _module(
        "llmrouter.utils", load_model=_load_model, save_model=_save_model, get_longformer_embedding=_unused,
        call_api=_unused, generate_task_query=_unused, calculate_task_performance=_unused)
    sys.modules["llmrouter.models.meta_router"] = _module("llmrouter.models.meta_router", MetaRouter=MetaRouter)
    return MetaRouter


def load_llmrouter(dest: Path) -> types.SimpleNamespace:
    MetaRouter = register_llmrouter(dest)
    mf = importlib.import_module("llmrouter.models.mfrouter")
    graph = importlib.import_module("llmrouter.models.graphrouter")
    hybrid = importlib.import_module("llmrouter.models.hybrid_llm")
    return types.SimpleNamespace(MetaRouter=MetaRouter, mf=mf, graph=graph, hybrid=hybrid)


def load_graphrouter(dest: Path) -> types.ModuleType:
    return _load_file("ev10_graphrouter_official", dest / "graphrouter" / "model" / "graph_nn.py")


class _NoLog:  # stands in for the wandb run the official trainer logs to
    def log(self, *args, **kwargs):
        pass


def load_irt(dest: Path) -> types.ModuleType:
    return _load_file("ev10_irt_mirt", dest / "irt_router" / "router" / "MIRT.py")


def load_avengers(dest: Path) -> types.SimpleNamespace:
    root = dest / "avengers_pro"

    def no_encoder(*args, **kwargs):
        raise KeyError("tokenizer is only used for remote embedding requests")

    class Dataset:  # imported by simple_cluster_router but only used by its data loader
        pass

    config = _load_file("ev10_avengers_config", root / "config.py")
    stubs = {
        "embedding_cache": _module("embedding_cache", EmbeddingCache=lambda **kwargs: None),
        "tiktoken": _module("tiktoken", encoding_for_model=no_encoder, get_encoding=lambda name: None),
        "datasets": _module("datasets", Dataset=Dataset),
        "config": config,
    }
    with temporary_modules(stubs):
        simple = _load_file("simple_cluster_router", root / "simple_cluster_router.py")
        balance = _load_file("ev10_avengers_balance", root / "balance_cluster_router.py")
        sys.modules.pop("simple_cluster_router", None)
    return types.SimpleNamespace(config=config, simple=simple, balance=balance)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------

def group_descriptions(groups: dict, service_cost: dict) -> dict:
    """One text per tier slot: its members in every source and its service price (plan 4.3)."""
    price = {"low": "cheapest", "medium": "middle", "high": "most expensive"}
    out = {}
    for tier in TIERS:
        members = [f"{SOURCE_NAMES[source]}: {', '.join(groups[source][tier])}" for source in SOURCES]
        out[tier] = (f"{tier} tier model group, {price[tier]} of three price tiers "
                     f"(relative service price {service_cost[tier]}). Members: " + "; ".join(members) + ".")
    return out


def model_descriptions(members: dict) -> dict:
    """GraphRouter model-node text per source, in pool order: name, tier and price relative to the pool's dearest."""
    out = {}
    for source in SOURCES:
        top = max(m["cost"] for m in members[source])
        out[source] = [f"{m['model']}: candidate model in the {SOURCE_NAMES[source]} pool, {m['tier']} price tier, "
                       f"relative price {m['cost'] / top:.3f}." for m in members[source]]
    return out


def fill_missing_with_target(success: pd.DataFrame, tiers: list, target: pd.DataFrame) -> tuple:
    """Fill a missing model record with the row's group success probability and check the group means.

    success: rows x models (0/1/NaN); tiers: tier of each column; target: rows x success_<tier> from 03_ours.
    Filling with the observed group mean keeps that mean; a fully missing group takes 03_ours's filled value.
    """
    filled = success.copy()
    for model, tier in zip(success.columns, tiers):
        filled[model] = success[model].fillna(target[f"success_{tier}"])
    for tier in TIERS:
        cols = [m for m, t in zip(success.columns, tiers) if t == tier]
        if not np.allclose(filled[cols].mean(axis=1).to_numpy(), target[f"success_{tier}"].to_numpy()):
            raise SystemExit(f"model outcomes do not reproduce the {tier} group probability")
    return filled, success.isna()


def tier_means(pred: np.ndarray, tiers: list) -> np.ndarray:
    """(questions x models) edge scores -> (questions x 3) mean over each tier's models."""
    return np.column_stack([pred[:, [i for i, t in enumerate(tiers) if t == tier]].mean(axis=1) for tier in TIERS])


def build_outcomes(run_dir: Path, repo: Path, data_root: Path, output_dir: Path) -> dict:
    """Per-model success of the 03_ours blended training rows, with the loaders 00_prepare used."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import prepare as P
    data_dir, prepare_dir = run_dir / "ours" / "data", run_dir / "prepare"
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    train_path = data_dir / "blended" / "train.parquet"
    if sha256_file(train_path) != data_manifest["variants"]["blended"]["files"]["train.parquet"]:
        raise SystemExit("03_ours training data changed")
    prepare_manifest = json.loads((prepare_dir / "manifest.json").read_text())
    cfg = prepare_manifest["config"]
    inputs = {k: (repo / v if (repo / v).exists() else data_root / v) for k, v in cfg["inputs"].items()}
    train = pd.read_parquet(train_path, columns=["sample_id", "source_dataset"] + [f"success_{t}" for t in TIERS])
    groups = json.loads((prepare_dir / "groups.json").read_text())
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("output directory must be new and empty")
    v2 = P.accepted_matrices(inputs["v2_data_config"], data_root, repo / "ml/src")
    records = dict(v2)
    for source, spec in cfg["external"].items():
        ext = pd.read_parquet(inputs[spec["input"]])
        ext = ext[ext.exclusion_reason.isna()].set_index("sample_id")
        records[source] = ext[[c for c in ext.columns if c.startswith("success_")]].rename(columns=lambda c: c[len("success_"):])
    members, frames, summary = {}, [], {}
    for source in SOURCES:
        pool = sorted((r for r in groups[source] if r["kept"]), key=lambda r: r["cost"])
        sealed = prepare_manifest["summary"]["groups"][source]
        if {t: sorted(r["model"] for r in pool if r["tier"] == t) for t in TIERS} != {t: sorted(sealed[t]) for t in TIERS}:
            raise SystemExit(f"groups.json differs from the sealed group list: {source}")
        members[source] = [{"model": r["model"], "tier": r["tier"], "cost": float(r["cost"])} for r in pool]
        rows = train[train.source_dataset.eq(source)].set_index("sample_id")
        names, tiers = [m["model"] for m in members[source]], [m["tier"] for m in members[source]]
        success = records[source].reindex(rows.index)[names].astype(float)
        filled, missing = fill_missing_with_target(success, tiers, rows)
        long = filled.reset_index().melt(id_vars="sample_id", var_name="model", value_name="success")
        long["filled"] = missing.reset_index().melt(id_vars="sample_id", value_name="m")["m"].to_numpy()
        frames.append(long.assign(source_dataset=source))
        summary[source] = {"rows": int(len(rows)), "models": len(names), "filled_cells": int(missing.to_numpy().sum())}
    output_dir.mkdir(parents=True, exist_ok=True)
    out = pd.concat(frames, ignore_index=True)[["sample_id", "source_dataset", "model", "success", "filled"]]
    out.to_parquet(output_dir / "train-model-outcomes.parquet", index=False)
    (output_dir / "members.json").write_text(json.dumps(members, ensure_ascii=False, indent=1))
    manifest = {"rows": summary, "check": "group means over models equal the 03_ours targets",
                "inputs": {"ours_data_manifest": sha256_file(data_dir / "manifest.json"),
                           "prepare_manifest": sha256_file(prepare_dir / "manifest.json"),
                           "groups.json": sha256_file(prepare_dir / "groups.json")},
                "outputs": {n: sha256_file(output_dir / n) for n in ("train-model-outcomes.parquet", "members.json")}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def encode_texts(texts: list, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=device)
    model.max_seq_length = 512
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype(np.float32)


def load_inputs(run_dir: Path, smoke: bool) -> dict:
    embed_dir, data_dir, prepare_dir = run_dir / "embed", run_dir / "ours" / "data", run_dir / "prepare"
    embed_manifest = json.loads((embed_dir / "manifest.json").read_text())
    for name, digest in embed_manifest["outputs"].items():
        if sha256_file(embed_dir / name) != digest:
            raise SystemExit(f"01_embed output changed: {name}")
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    for variant in ("blended", "a_only"):
        for name, digest in data_manifest["variants"][variant]["files"].items():
            if sha256_file(data_dir / variant / name) != digest:
                raise SystemExit(f"03_ours data changed: {variant}/{name}")
    if sha256_file(prepare_dir / "manifest.json") != embed_manifest["prepare_manifest_sha256"]:
        raise SystemExit("00_prepare manifest differs from the one 01_embed used")
    prepare_manifest = json.loads((prepare_dir / "manifest.json").read_text())

    ids = pd.read_parquet(embed_dir / "ids.parquet")
    vectors = np.load(embed_dir / "embeddings.npy").astype(np.float32)
    columns = ["sample_id", "source_dataset", "sample_weight"] + [f"success_{t}" for t in TIERS]
    train = {v: pd.read_parquet(data_dir / v / "train.parquet", columns=columns) for v in ("blended", "a_only")}
    scored = ids[ids.role.isin(SCORED_ROLES)].reset_index(drop=True)
    if smoke:
        train = {v: f.groupby("source_dataset", group_keys=False).head(600).reset_index(drop=True) for v, f in train.items()}
        scored = scored.groupby("role", group_keys=False).head(200).reset_index(drop=True)
    row = pd.Series(np.arange(len(ids)), index=ids.sample_id)
    train_roles = ids.set_index("sample_id").role
    for variant, frame in train.items():
        if not train_roles.reindex(frame.sample_id).eq("train").all():
            raise SystemExit(f"{variant} training rows must all be 01_embed train rows")
    groups = prepare_manifest["summary"]["groups"]  # {source: {tier: [models]}}, sealed with the manifest
    service_cost = prepare_manifest["config"]["service_cost"]
    hashes = {"embed_manifest": sha256_file(embed_dir / "manifest.json"),
              "ours_data_manifest": sha256_file(data_dir / "manifest.json"),
              "prepare_manifest": sha256_file(prepare_dir / "manifest.json")}
    outcomes_dir, model_outcomes, members = run_dir / "common-inputs", None, None
    if (outcomes_dir / "manifest.json").exists():  # written by build-outcomes; needed by GraphRouter impl=official
        om = json.loads((outcomes_dir / "manifest.json").read_text())
        for name, digest in om["outputs"].items():
            if sha256_file(outcomes_dir / name) != digest:
                raise SystemExit(f"common-inputs changed: {name}")
        if om["inputs"]["ours_data_manifest"] != hashes["ours_data_manifest"]:
            raise SystemExit("common-inputs were built from different 03_ours data")
        members = json.loads((outcomes_dir / "members.json").read_text())
        model_outcomes = pd.read_parquet(outcomes_dir / "train-model-outcomes.parquet")
        model_outcomes = model_outcomes[model_outcomes.sample_id.isin(train["blended"].sample_id)]
        hashes["common_inputs_manifest"] = sha256_file(outcomes_dir / "manifest.json")
    return {
        "vectors": vectors, "row": row, "train": train, "scored": scored, "groups": groups, "smoke": smoke,
        "service_cost": {t: float(service_cost[t]) for t in TIERS}, "members": members,
        "model_outcomes": model_outcomes, "hashes": hashes,
    }


class Context:
    def __init__(self, inputs: dict, description_vectors: np.ndarray, device: str):
        self.inputs = inputs
        self.device = device
        self.llm_vectors = description_vectors  # (3, 1024), tier order
        self.model_vectors = {}  # source -> (models, 1024), pool order; set by run() for GraphRouter impl=official
        self.smoke = inputs.get("smoke", False)
        self.scored = inputs["scored"]
        self.query = inputs["vectors"][inputs["row"].loc[self.scored.sample_id].to_numpy()]

    def train(self, variant: str = "blended"):
        frame = self.inputs["train"][variant]
        x = self.inputs["vectors"][self.inputs["row"].loc[frame.sample_id].to_numpy()]
        y = frame[[f"success_{t}" for t in TIERS]].to_numpy(np.float64)
        if np.isnan(y).any():
            raise SystemExit("training targets must be complete (03_ours fills single missing tiers)")
        return frame, x, y

    def routing_frame(self, variant: str = "blended") -> pd.DataFrame:
        """LLMRouter long format, tiers in cost order so that a tie picks the cheaper tier."""
        frame, _, y = self.train(variant)
        n = len(frame)
        return pd.DataFrame({
            "query": np.repeat(frame.sample_id.to_numpy(), 3),
            "model_name": np.tile(np.array(TIERS), n),
            "performance": y.reshape(-1),
            "embedding_id": np.repeat(np.arange(n), 3),
        })


# ---------------------------------------------------------------------------------------------
# Methods: each yields (setting, scores (n, 3), extra columns, info) for the settings still to run
# ---------------------------------------------------------------------------------------------

def routerbench_frame(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"prompt": frame.sample_id.to_numpy()})
    for k, tier in enumerate(TIERS):
        out[tier] = y[:, k]
        out[f"{tier}|model_response"] = "-"  # read only for willingness_to_pay costs (unused)
    return out


def run_knn(ctx: Context, ext, settings, variant="blended"):
    frame, x, y = ctx.train(variant)
    cache = ArrayCache(frame.sample_id, x)
    for setting in settings:
        start = time.time()
        router = ext.routerbench.KNNRouter(embedding_model=EMBEDDING_MODEL, cache=cache, train_file=routerbench_frame(frame, y),
                                           models_to_route=list(TIERS), **setting)
        fit = time.time() - start
        start = time.time()
        parts = [router.calc_performance_scores(ctx.query[i:i + 2048])[list(TIERS)].to_numpy()
                 for i in range(0, len(ctx.query), 2048)]
        yield setting, np.vstack(parts), {}, {"fit_seconds": fit, "score_seconds": time.time() - start, "train_rows": len(frame)}


def run_routerbench_regressor(kind: str):
    def runner(ctx: Context, ext, settings):
        frame, x, y = ctx.train()
        cache = ArrayCache(frame.sample_id, x)
        cls = {"mlp": ext.routerbench.MLPRouter, "svm": ext.routerbench.SVMRouter}[kind]
        for setting in settings:
            start = time.time()
            router = cls(embedding_model=EMBEDDING_MODEL, cache=cache, train_file=routerbench_frame(frame, y),
                         models_to_route=list(TIERS), **setting)
            fit = time.time() - start
            start = time.time()
            scores = router.get_model_to_route_to(ctx.query)
            yield setting, np.column_stack([scores[t] for t in TIERS]), {}, {
                "fit_seconds": fit, "score_seconds": time.time() - start, "train_rows": len(frame)}
    return runner


def run_mf(ctx: Context, ext, settings, out_dir: Path):
    import torch
    frame, x, _ = ctx.train()
    for setting in settings:
        seed_everything()
        ext.llmrouter.MetaRouter.context = {
            "cfg": {"hparam": {"latent_dim": 128, "text_dim": x.shape[1], "lr": 0.001, "epochs": setting["epochs"],
                               "batch_size": 64, "noise_alpha": 0.0},
                    "model_path": {"save_model_path": str(out_dir / setting_id("mf", setting) / "mf-model.pt")}},
            "routing_data_train": ctx.routing_frame(),
            "query_embedding_data": torch.from_numpy(x),
        }
        start = time.time()
        router = ext.llmrouter.mf.MFRouter(yaml_path=None)
        trainer = ext.llmrouter.mf.MFRouterTrainer(router, device=ctx.device)
        trainer.train()
        fit = time.time() - start
        model = trainer.model.eval()
        start = time.time()
        with torch.no_grad():
            query = torch.from_numpy(ctx.query).to(ctx.device)
            scores = np.vstack([model.score_all(model.project_text(q)).cpu().numpy() for q in query])
        info = {"fit_seconds": fit, "score_seconds": time.time() - start, "train_rows": len(frame),
                "pairs": len(router.pairs), "parameters": sum(p.numel() for p in model.parameters()),
                "tier_order": [router.idx_to_model[i] for i in range(3)]}
        if info["tier_order"] != list(TIERS):
            raise SystemExit("MF model index order must be low, medium, high")
        yield setting, scores, {}, info


def run_graphrouter(ctx: Context, ext, settings, out_dir: Path):
    official = [st for st in settings if st["impl"] == "official"]
    if any(st["impl"] == "llmrouter" for st in settings):
        yield from run_graphrouter_llmrouter(ctx, ext, out_dir)
    if official:
        yield from run_graphrouter_official(ctx, ext, official, out_dir)


def run_graphrouter_llmrouter(ctx: Context, ext, out_dir: Path):
    import torch
    from sklearn.preprocessing import MinMaxScaler
    frame, x, _ = ctx.train()
    folder = out_dir / "training-llmrouter"
    seed_everything()
    routing = ctx.routing_frame()
    routing["query"] = routing["query"].astype("category")  # same values, faster row lookups in the original loops
    ext.llmrouter.MetaRouter.context = {
        "cfg": {"hparam": {"hidden_dim": 64, "learning_rate": 0.001, "weight_decay": 0.0001, "train_epoch": 100,
                           "batch_size": 4, "train_mask_rate": 0.3, "val_split_ratio": 0.2, "random_state": SEED},
                "model_path": {"save_model_path": str(folder / "graphrouter.pt"),
                               "ini_model_path": str(folder / "graphrouter-init-unused.pt")},
                "llm_data": {t: {"embedding": ctx.llm_vectors[k].tolist()} for k, t in enumerate(TIERS)}},
        "routing_data_train": routing,
        "query_embedding_data": torch.from_numpy(x),
    }
    folder.mkdir(parents=True, exist_ok=True)
    start = time.time()
    router = ext.llmrouter.graph.GraphRouter(yaml_path=None)
    if router.model_names != list(TIERS):
        raise SystemExit("GraphRouter LLM order must be low, medium, high")
    best = ext.llmrouter.graph.GraphTrainer(router, device=router.device).train()
    fit = time.time() - start
    predictor = router.gnn_predictor
    predictor.model.load_state_dict(torch.load(folder / "graphrouter.pt", map_location=router.device))
    start = time.time()
    scaler = MinMaxScaler().fit(x)  # the scaler _prepare_training_data fitted on the training questions
    new = scaler.transform(ctx.query)
    n_train, n_new = len(router.query_embedding_list), len(new)
    queries = np.vstack([router.query_embedding_list, new])
    edges = (n_train + n_new) * 3
    mask_train = torch.zeros(edges)
    mask_train[: n_train * 3] = 1
    mask_new = 1 - mask_train
    performance = np.concatenate([router.performance_list, np.zeros(n_new * 3)])
    label = np.eye(3)[np.argmax(performance.reshape(-1, 3), axis=1)].reshape(-1, 1)
    data = router.form_data.formulation(
        query_feature=queries, llm_feature=router.llm_embedding,
        org_node=[q for q in range(n_train + n_new) for _ in range(3)], des_node=[0, 1, 2] * (n_train + n_new),
        edge_feature=performance, label=label, edge_mask=mask_new, train_mask=mask_train,
        valide_mask=torch.zeros(edges), test_mask=mask_new)
    predictor.model.eval()
    with torch.no_grad():  # GNNPredictor.predict without the final argmax
        edge_predict = predictor.model(
            query_features=data.query_features, llm_features=data.llm_features, edge_index=data.edge_index,
            edge_mask=data.edge_mask.clone().to(torch.bool),
            edge_can_see=torch.logical_or(data.train_mask, data.valide_mask).to(torch.bool),
            edge_weight=data.edge_attr)
    yield {"impl": "llmrouter"}, edge_predict.reshape(-1, 3).cpu().numpy(), {}, {
        "fit_seconds": fit, "score_seconds": time.time() - start, "train_rows": len(frame),
        "best_validation_result": float(best), "variant": "original", "external_key": "llmrouter",
        "parameters": sum(p.numel() for p in predictor.model.parameters())}


def official_graph(form, query: np.ndarray, llm: np.ndarray, effect: np.ndarray, cost: np.ndarray, label: np.ndarray,
                   edge_mask, train_mask, valide_mask, test_mask):
    """The official form_data.formulation call: question nodes x model nodes, 3-d edges [effect, cost, effect]."""
    n, models = len(query), len(llm)
    return form.formulation(task_id=np.zeros((n, llm.shape[1]), np.float32), query_feature=query, llm_feature=llm,
                            org_node=[i for i in range(n) for _ in range(models)], des_node=list(range(models)) * n,
                            edge_feature=effect, label=label, edge_mask=edge_mask,
                            combined_edge=np.stack([np.tile(cost, n), effect], axis=1),
                            train_mask=train_mask, valide_mask=valide_mask, test_mask=test_mask)


def run_graphrouter_official(ctx: Context, ext, settings, out_dir: Path):
    import torch
    if ctx.inputs.get("model_outcomes") is None:
        raise SystemExit("GraphRouter impl=official needs <run>/common-inputs from `common_track.py build-outcomes`")
    gr = ext.graphrouter
    frame, _, _ = ctx.train()
    outcomes, members = ctx.inputs["model_outcomes"], ctx.inputs["members"]
    vectors, row = ctx.inputs["vectors"], ctx.inputs["row"]
    scored_source = ctx.scored.source_dataset.to_numpy()
    results = {setting_id("graphrouter", st): {"scores": np.full((len(ctx.scored), 3), np.nan), "sources": {},
                                              "fit": 0.0, "score": 0.0} for st in settings}

    def mask(n, models, questions):
        m = torch.zeros(n * models)
        m[np.repeat(questions, models) * models + np.tile(np.arange(models), len(questions))] = 1
        return m

    for label_kind, epochs in dict.fromkeys((st["label"], st["epochs"]) for st in settings):
        group = [st for st in settings if (st["label"], st["epochs"]) == (label_kind, epochs)]
        for source in SOURCES:
            pool = members[source]
            names, tiers = [m["model"] for m in pool], [m["tier"] for m in pool]
            cost = np.array([m["cost"] for m in pool], float)
            cost = cost / cost.max()
            wide = outcomes[outcomes.source_dataset.eq(source)].pivot(index="sample_id", columns="model", values="success")
            wide = wide.reindex(frame.loc[frame.source_dataset.eq(source), "sample_id"])[names]
            if wide.isna().any().any():
                raise SystemExit(f"model outcomes must cover every training row: {source}")
            q_train, perf, llm = vectors[row.loc[wide.index].to_numpy()], wide.to_numpy(float), ctx.model_vectors[source]
            n_train, models = perf.shape
            seed_everything()
            order = np.random.permutation(n_train)
            val_q, fit_q = order[: n_train // 8], order[n_train // 8:]  # official split .7/.1 of the labelled part
            effect = perf.reshape(-1)  # Performance First: 1.0 x effect - 0.0 x cost
            target = np.eye(models)[np.argmax(perf, axis=1)].reshape(-1, 1) if label_kind == "best" else effect.reshape(-1, 1).copy()
            m_fit, m_val, none = mask(n_train, models, fit_q), mask(n_train, models, val_q), torch.zeros(n_train * models)
            form = gr.form_data(ctx.device)
            data_fit = official_graph(form, q_train, llm, effect, cost, target, m_fit, m_fit, m_val, none)
            data_val = official_graph(form, q_train, llm, effect, cost, target, m_val, m_fit, m_val, none)
            folder = out_dir / f"training-official-{label_kind}-{epochs}" / source
            folder.mkdir(parents=True, exist_ok=True)
            config = {"learning_rate": 3e-4, "weight_decay": 1e-4, "model_path": str(folder / "best_f1.pth"),
                      "train_epoch": min(epochs, 3) if ctx.smoke else epochs, "batch_size": 32, "train_mask_rate": 0.5,
                      "llm_num": models}
            predictor = gr.GNN_prediction(query_feature_dim=q_train.shape[1], llm_feature_dim=llm.shape[1],
                                          hidden_features_size=8, in_edges_size=3, wandb=_NoLog(), config=config,
                                          device=ctx.device)
            losses, criterion = [], predictor.criterion

            def recorded(pred, y, criterion=criterion, losses=losses):
                loss = criterion(pred, y)
                losses.append(float(loss))
                return loss
            predictor.criterion = recorded
            start = time.time()
            # data_for_test only feeds the per-epoch log line; it is our validation split, never dev or test
            predictor.train_validate(data=data_fit, data_validate=data_val, data_for_test=data_val)
            fit = time.time() - start
            per_epoch = np.array(losses).reshape(config["train_epoch"], -1)[:, : config["batch_size"]].mean(axis=1)
            weights = {"last": {k: v.detach().clone() for k, v in predictor.model.state_dict().items()},
                       "best_f1": torch.load(folder / "best_f1.pth", map_location=ctx.device)}
            rows_new = np.where(scored_source == source)[0]
            query = np.vstack([q_train, ctx.query[rows_new]])
            n = len(query)
            seen = mask(n, models, np.arange(n_train))
            new = 1 - seen
            data = official_graph(form, query, llm, np.concatenate([effect, np.zeros(len(rows_new) * models)]), cost,
                                  np.zeros((n * models, 1)), new, seen, torch.zeros(n * models), new)
            for st in group:
                start = time.time()
                predictor.model.load_state_dict(weights[st["checkpoint"]])
                predictor.model.eval()
                with torch.no_grad():  # GNN_prediction.test: training and validation edges visible
                    pred = predictor.model(task_id=data.task_id, query_features=data.query_features,
                                           llm_features=data.llm_features, edge_index=data.edge_index,
                                           edge_mask=new.bool(), edge_can_see=seen.bool(), edge_weight=data.combined_edge)
                result = results[setting_id("graphrouter", st)]
                result["scores"][rows_new] = tier_means(pred.reshape(-1, models).cpu().numpy(), tiers)
                result["fit"] += fit
                result["score"] += time.time() - start
                result["sources"][source] = {"train_questions": int(len(fit_q)), "validation_questions": int(len(val_q)),
                                             "models": models, "fit_seconds": round(fit, 1),
                                             "train_loss_first_last_epoch": [round(float(per_epoch[0]), 4),
                                                                             round(float(per_epoch[-1]), 4)],
                                             "parameters": sum(p.numel() for p in predictor.model.parameters())}
    for st in settings:
        result = results[setting_id("graphrouter", st)]
        yield st, result["scores"], {}, {
            "fit_seconds": result["fit"], "score_seconds": result["score"], "train_rows": len(frame),
            "variant": "original (paper label)" if st["label"] == "best" else "partial borrowing (success label)",
            "external_key": "graphrouter", "diagnostic_unscored": True, "shared_training_per_label_and_epochs": True,
            "sources": result["sources"]}


def run_hybrid(ctx: Context, ext, settings, out_dir: Path):
    import torch
    frame, x, _ = ctx.train()
    for setting in settings:
        folder = out_dir / setting_id("hybrid_llm", setting)
        seed_everything()
        ext.llmrouter.MetaRouter.context = {
            "cfg": {"router_mode": setting["router_mode"], "router_tau": 0.1, "router_threshold": 0.5,
                    "hparam": {"hidden_layer_sizes": [128, 64], "activation": "relu", "solver": "adam", "max_iter": 300,
                               "random_state": SEED},
                    "model_path": {"ini_model_path": str(folder / "hybrid-init-unused.pkl"),
                                   "save_model_path": str(folder / "hybrid.pkl")}},
            "routing_data_train": ctx.routing_frame(),
            "query_embedding_data": torch.from_numpy(x),
            "llm_data": {"low": {"size": "1B"}, "medium": {"size": "2B"}, "high": {"size": "3B"}},
        }
        start = time.time()
        router = ext.llmrouter.hybrid.HybridLLMRouter(yaml_path=None)
        if (router.small_model_name, router.large_model_name) != ("low", "high"):
            raise SystemExit("Hybrid LLM must compare the low and high groups")
        trainer = ext.llmrouter.hybrid.HybridLLMTrainer(router)
        trainer.train()
        fit = time.time() - start
        start = time.time()
        score = np.clip(trainer.model.predict(ctx.query), 0.0, 1.0)  # route_single: clip to [0, 1]
        yield setting, np.repeat(score[:, None], 3, axis=1), {}, {
            "fit_seconds": fit, "score_seconds": time.time() - start, "train_rows": len(frame), "single_score": True,
            "label_mean": float(np.mean(router.router_label_list))}


def cold_start_vectors(train_x: np.ndarray, query: np.ndarray, k: int = 5) -> np.ndarray:
    """IRT-Router utils/cold_test.py: mean embedding of the k nearest training questions (Euclidean)."""
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(train_x)
    _, index = knn.kneighbors(query)
    return train_x[index].mean(axis=1)


def run_irt(ctx: Context, ext, settings):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    frame, x, y = ctx.train()
    seed_everything()
    llm = torch.from_numpy(np.tile(ctx.llm_vectors, (len(frame), 1)))
    item = torch.from_numpy(np.repeat(x, 3, axis=0))
    response = torch.tensor(y.reshape(-1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(llm, item, response), batch_size=512, shuffle=True)
    start = time.time()
    cdm = ext.irt.MIRT(ctx.llm_vectors.shape[1], x.shape[1], 25)
    cdm.train(loader, None, epoch=9, device=ctx.device, lr=0.001)
    fit = time.time() - start
    cold = cold_start_vectors(x, ctx.query)
    for setting in settings:
        start = time.time()
        q = (1 - setting["lamda"]) * ctx.query + setting["lamda"] * cold
        scores, difficulty = np.zeros((len(q), 3)), None
        with torch.no_grad():
            for k in range(3):
                l_vec = torch.from_numpy(np.repeat(ctx.llm_vectors[k:k + 1], len(q), axis=0))
                scores[:, k] = cdm.generate(l_vec, torch.from_numpy(q.astype(np.float32)), device=ctx.device)
                if k == 0:
                    difficulty = np.asarray(cdm.get_difficulty(l_vec, torch.from_numpy(q.astype(np.float32)), device=ctx.device))
        yield setting, scores, {"irt_difficulty": difficulty}, {
            "fit_seconds": fit, "score_seconds": time.time() - start, "train_rows": len(frame),
            "parameters": sum(p.numel() for p in cdm.irt_net.parameters())}


def avengers_scores(router, query: np.ndarray, models: list) -> np.ndarray:
    """SimpleClusterRouter.route_queries_batch scoring, returning every model's score."""
    q = router.normalizer.transform(query)
    distances = 1 - q @ router.cluster_centers.T
    out = np.zeros((len(q), len(models)))
    for i in range(len(q)):
        closest = np.argsort(distances[i])[: router.config.top_k]
        logits = -router.config.beta * distances[i][closest]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        for cluster, prob in zip(closest, probs):
            if cluster not in router.cluster_rankings:
                continue
            ranking = router.cluster_rankings[cluster]["ranking"]
            for m, name in enumerate(models):
                if name in ranking:
                    out[i, m] += prob * (1.0 / (ranking.index(name) + 1))
    return out


def run_avengers(ctx: Context, ext, settings, data_path: Path):
    frame, x, y = ctx.train()
    cost = ctx.inputs["service_cost"]
    records = [{"query": sid, "records": {t: float(y[i, k]) for k, t in enumerate(TIERS)},
                "usages": {t: {"cost": cost[t]} for t in TIERS}} for i, sid in enumerate(frame.sample_id)]
    lookup = dict(zip(frame.sample_id, x))
    for setting in settings:
        seed_everything()
        config = ext.avengers.config.SimpleClusterConfig(
            data_path=str(data_path), seed=SEED, n_clusters=setting["n_clusters"], top_k=3, beta=9.0, max_router=1,
            performance_weight=setting["performance_weight"], cost_sensitivity=round(1 - setting["performance_weight"], 10),
            embedding_model=EMBEDDING_MODEL)
        router = ext.avengers.balance.BalanceClusterRouter(config)
        router.available_models = list(TIERS)
        router._generate_embeddings_concurrent = lambda queries: [lookup[q] for q in queries]
        start = time.time()
        router.build_cluster_model(records)
        fit = time.time() - start
        start = time.time()
        scores = avengers_scores(router, ctx.query, list(TIERS))
        yield setting, scores, {}, {"fit_seconds": fit, "score_seconds": time.time() - start, "train_rows": len(frame),
                                    "rankings": {int(c): r["ranking"] for c, r in router.cluster_rankings.items()}}


def run_r1_bge(ctx: Context, ext, settings):
    import torch
    from torch import nn
    frame, x, y = ctx.train()
    weight = frame.sample_weight.to_numpy(np.float32)
    for setting in settings:
        seed_everything()
        model = nn.Sequential(nn.Linear(x.shape[1], setting["hidden"]), nn.GELU(), nn.Dropout(0.1),
                              nn.Linear(setting["hidden"], 3)).to(ctx.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
        wt = torch.from_numpy(weight if setting["source_weight"] else np.ones_like(weight))
        generator = torch.Generator().manual_seed(SEED)
        start = time.time()
        model.train()
        for _ in range(setting["epochs"]):
            for idx in torch.randperm(len(xt), generator=generator).split(256):
                xb, yb, wb = xt[idx].to(ctx.device), yt[idx].to(ctx.device), wt[idx].to(ctx.device)
                loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb, reduction="none").mean(1)
                loss = (loss * wb).sum() / wb.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        fit = time.time() - start
        model.eval()
        start = time.time()
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.from_numpy(ctx.query).to(ctx.device))).cpu().numpy()
        yield setting, scores, {}, {"fit_seconds": fit, "score_seconds": time.time() - start, "train_rows": len(frame),
                                    "parameters": sum(p.numel() for p in model.parameters())}


METHODS = ("knn", "knn_a", "mlp", "svm", "mf", "graphrouter", "hybrid_llm", "irt_router", "avengers_pro", "r1_bge")
EXTERNAL_OF = {"knn": "routerbench", "knn_a": "routerbench", "mlp": "routerbench", "svm": "routerbench",
               "mf": "llmrouter", "graphrouter": "graphrouter", "hybrid_llm": "llmrouter", "irt_router": "irt_router",
               "avengers_pro": "avengers_pro", "r1_bge": None}


def runner_for(method: str, ctx: Context, ext, settings, out_dir: Path, data_path: Path):
    if method == "knn":
        return run_knn(ctx, ext, settings)
    if method == "knn_a":
        return run_knn(ctx, ext, settings, variant="a_only")
    if method in ("mlp", "svm"):
        return run_routerbench_regressor(method)(ctx, ext, settings)
    if method == "mf":
        return run_mf(ctx, ext, settings, out_dir / method)
    if method == "graphrouter":
        return run_graphrouter(ctx, ext, settings, out_dir / method)
    if method == "hybrid_llm":
        return run_hybrid(ctx, ext, settings, out_dir / method)
    if method == "irt_router":
        return run_irt(ctx, ext, settings)
    if method == "avengers_pro":
        return run_avengers(ctx, ext, settings, data_path)
    return run_r1_bge(ctx, ext, settings)


def write_setting(folder: Path, ctx: Context, method: str, setting: dict, scores: np.ndarray, extra: dict, info: dict,
                  deviations: list | None = None, external: dict | None = None):
    info = dict(info)
    external_key = info.pop("external_key", EXTERNAL_OF.get(method))
    external = EXTERNAL if external is None else external
    pooled = ctx.scored.role.ne("diagnostic").to_numpy()  # dev and test rows must always be scored
    if scores.shape != (len(ctx.scored), 3) or not np.isfinite(scores[pooled]).all() or \
            not (np.isfinite(scores[~pooled]).all() or info.get("diagnostic_unscored")):
        raise SystemExit(f"{method} {setting}: scores must be finite with shape {(len(ctx.scored), 3)}")
    folder.mkdir(parents=True, exist_ok=True)
    table = ctx.scored[["sample_id", "source_dataset", "role"]].copy()
    for k, tier in enumerate(TIERS):
        table[f"score_{tier}"] = scores[:, k].astype(np.float64)
    for name, values in extra.items():
        table[name] = np.asarray(values, dtype=np.float64)
    table.to_parquet(folder / "scores.parquet", index=False)
    record = {"method": method, "setting_id": folder.name, "setting": setting, "seed": SEED,
              "external": external_key and {k: external[external_key][k] for k in ("repo", "commit", "license")},
              "deviations": DEVIATIONS["all"] + DEVIATIONS[method] if deviations is None else deviations,
              "device": ctx.device,
              "score_rows": int(len(table)), "score_ms_per_1k": 1e6 * info["score_seconds"] / len(table),
              # plan 4.5: an original that does not work on our data is reported as a reproduction failure
              "degenerate": bool((scores[pooled].std(axis=0) < 1e-6).all()),
              "scores_sha256": sha256_file(folder / "scores.parquet")} | info
    (folder / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    (folder / "_SUCCESS").write_text("")
    return record


def run(run_dir: Path, external_dir: Path, output_dir: Path, methods: list, device: str, smoke: bool) -> dict:
    import sklearn
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    external = verify_external(external_dir)
    inputs = load_inputs(run_dir, smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptions = group_descriptions(inputs["groups"], inputs["service_cost"])
    desc_path = output_dir / "group-descriptions.json"
    if desc_path.exists():
        saved = json.loads(desc_path.read_text())
        if saved["texts"] != descriptions:
            raise SystemExit("group descriptions changed since the last run in this folder")
        description_vectors = np.asarray(saved["vectors"], dtype=np.float32)
    else:
        description_vectors = encode_texts([descriptions[t] for t in TIERS], device)
        desc_path.write_text(json.dumps({"model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION, "texts": descriptions,
                                         "vectors": description_vectors.tolist()}, ensure_ascii=False, indent=1))
    ctx = Context(inputs, description_vectors, device)
    model_desc_path = output_dir / "model-descriptions.json"
    if inputs["members"] is not None:
        texts = model_descriptions(inputs["members"])
        if model_desc_path.exists():
            saved = json.loads(model_desc_path.read_text())
            if saved["texts"] != texts:
                raise SystemExit("model descriptions changed since the last run in this folder")
        else:
            flat = [t for source in SOURCES for t in texts[source]]
            vectors = encode_texts(flat, device).tolist()
            saved, k = {"model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION, "texts": texts, "vectors": {}}, 0
            for source in SOURCES:
                saved["vectors"][source] = vectors[k: k + len(texts[source])]
                k += len(texts[source])
            model_desc_path.write_text(json.dumps(saved, ensure_ascii=False, indent=1))
        ctx.model_vectors = {source: np.asarray(v, dtype=np.float32) for source, v in saved["vectors"].items()}
    ext = types.SimpleNamespace(routerbench=load_routerbench(external_dir), llmrouter=load_llmrouter(external_dir),
                                graphrouter=load_graphrouter(external_dir), irt=load_irt(external_dir),
                                avengers=load_avengers(external_dir))
    data_path = run_dir / "ours" / "data" / "blended" / "train.parquet"
    for method in methods:
        todo = [s for s in SETTINGS[method] if not (output_dir / method / setting_id(method, s) / "_SUCCESS").exists()]
        print(f"[{method}] {len(SETTINGS[method]) - len(todo)} done, {len(todo)} to run", flush=True)
        if not todo:
            continue
        for setting, scores, extra, info in runner_for(method, ctx, ext, todo, output_dir, data_path):
            record = write_setting(output_dir / method / setting_id(method, setting), ctx, method, setting, scores, extra, info)
            print(f"[{method}] {record['setting_id']}: fit {info['fit_seconds']:.0f}s, score {info['score_seconds']:.0f}s", flush=True)
    outputs = {}
    for method in METHODS:
        for setting in SETTINGS[method]:
            folder = output_dir / method / setting_id(method, setting)
            if (folder / "_SUCCESS").exists():
                outputs[f"{method}/{folder.name}"] = json.loads((folder / "metrics.json").read_text())["scores_sha256"]
    manifest = {
        "experiment": "EV-10 04_common_track", "smoke": smoke, "seed": SEED, "device": device, "inputs": inputs["hashes"],
        "scored_rows": inputs["scored"].groupby(["role", "source_dataset"]).size().unstack(fill_value=0).to_dict("index"),
        "train_rows": {v: f.groupby("source_dataset").size().to_dict() for v, f in inputs["train"].items()},
        "external": external, "settings": {m: [setting_id(m, s) for s in SETTINGS[m]] for m in METHODS},
        "group_descriptions_sha256": sha256_file(desc_path), "deviations": DEVIATIONS,
        "model_descriptions_sha256": sha256_file(model_desc_path) if model_desc_path.exists() else None,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "torch_geometric": importlib.import_module("torch_geometric").__version__},
        "outputs": outputs, "complete": len(outputs) == sum(len(SETTINGS[m]) for m in METHODS),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return {k: manifest[k] for k in ("complete", "outputs")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--external-dir", type=Path, required=True)
    b = sub.add_parser("build-outcomes", help="local: per-model success of the training rows (GraphRouter model nodes)")
    b.add_argument("--run-dir", type=Path, required=True)
    b.add_argument("--repo", type=Path, required=True, help="checkout holding ml/src and committed inputs")
    b.add_argument("--data-root", type=Path, default=None, help="root for git-ignored data (defaults to --repo)")
    b.add_argument("--output-dir", type=Path, required=True, help="normally <run-dir>/common-inputs")
    r = sub.add_parser("run")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--external-dir", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    r.add_argument("--device", default="auto")
    r.add_argument("--smoke", action="store_true", help="600 training rows per source, 200 scored rows per role")
    a = p.parse_args(argv)
    if a.command == "fetch":
        out = fetch(a.external_dir)
    elif a.command == "build-outcomes":
        out = build_outcomes(a.run_dir, a.repo, a.data_root or a.repo, a.output_dir)
    else:
        out = run(a.run_dir, a.external_dir, a.output_dir, a.methods, a.device, a.smoke)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
