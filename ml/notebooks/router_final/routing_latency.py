"""EV-10 08a_latency: routing latency of every selected router, one question at a time, on one machine.

Routing latency = time from the question text to the start tier, in three parts:
  encoder   tokenisation and encoder (the shared BGE for the 04 routers, the router's own for the others)
  body      the router itself (neighbour search, regressor, graph, cluster scores, ...)
  decision  per-tier isotonic calibration and the escalation-aware rule (the same for every router)
Each router answers 200 fixed dev questions (50 per source) after 20 warm-up questions; median and p95 are
reported. For the BGE routers two totals are kept: with the encoder, and the part added when the BGE vector
already exists (the service embeds every question for its cache before routing).

Latency depends on the architecture and the data a router searches, not on trained weight values. Routers
whose weights were not kept (04 bodies, 05 encoders) are rebuilt with the same classes, settings and data
sizes and fitted only briefly; the trained ModernBERT and released RouteLLM weights are used as they are.
Plan 4.3: a reported metric, used only to break ties within +-.01 of utility. Colab timings are relative,
not the service's FastAPI latency. No outcomes are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import common_track as C  # noqa: E402
import finetune_track as F  # noqa: E402
import lock as L  # noqa: E402

PER_SOURCE, WARMUP = 50, 20
MODERNBERT = {"r0_group": "ours/models/r0_group/model", "r1_group": "ours/models/r1_group/model",
              "r1_group_a": "ours/models/r1_group_a/model", "rt08": "rt08/model/full/final-model"}


def sample_questions(inputs: dict, per_source: int = PER_SOURCE) -> pd.DataFrame:
    dev = inputs["scored"].assign(text=inputs["texts"], row=np.arange(len(inputs["scored"])))
    dev = dev[dev.role.eq("dev")].copy()
    dev["key"] = [hashlib.sha1(s.encode()).hexdigest() for s in dev.sample_id]
    return dev.sort_values("key").groupby("source_dataset", group_keys=False).head(per_source).reset_index(drop=True)


def stats(ms: list) -> dict:
    a = np.asarray(ms)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}


def timed(fn, items: list, sync=None) -> list:
    for item in items[:WARMUP]:
        fn(item)
    out = []
    for item in items:
        start = time.perf_counter()
        fn(item)
        if sync:
            sync()
        out.append(1000 * (time.perf_counter() - start))
    return out


class DecisionStage:
    """Isotonic calibration of three tier features plus the escalation-aware rule, for one question."""

    def __init__(self, n: int = 3000):
        from sklearn.isotonic import IsotonicRegression
        rng = np.random.default_rng(0)
        x = rng.uniform(size=(n, 3))
        self.iso = [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x[:, j], (rng.uniform(size=n) < x[:, j]).astype(float))
                    for j in range(3)]

    def __call__(self, features: np.ndarray) -> int:
        p = np.column_stack([iso.predict(features[:, j]) for j, iso in enumerate(self.iso)])
        return int(L.decide("escalation", p, 0.1, 0.3, L.predicted_chain(p, 0.5))[0])


# ---------------------------------------------------------------------------------------------
# Router bodies on a BGE vector (rebuilt with the 04 classes and settings)
# ---------------------------------------------------------------------------------------------

def bge_bodies(ctx, ext, settings: dict, sample: pd.DataFrame, device: str) -> dict:
    """method -> callable(row index in ctx.query) for the selected 04 settings."""
    import torch
    frame, x, y = ctx.train()
    bodies = {}
    rb_frame = C.routerbench_frame(frame, y)
    if "knn" in settings or "knn_a" in settings:
        for method in ("knn", "knn_a"):
            if method not in settings:
                continue
            f_, x_, y_ = ctx.train("a_only" if method == "knn_a" else "blended")
            router = ext.routerbench.KNNRouter(embedding_model=C.EMBEDDING_MODEL, cache=C.ArrayCache(f_.sample_id, x_),
                                               train_file=C.routerbench_frame(f_, y_), models_to_route=list(C.TIERS),
                                               **settings[method])
            bodies[method] = (lambda r: lambda i: r.calc_performance_scores(ctx.query[i:i + 1]))(router)
    small = slice(0, 2000)  # regressors: same architecture, briefly fitted
    cache = C.ArrayCache(frame.sample_id[small], x[small])
    if "mlp" in settings:
        r = ext.routerbench.MLPRouter(embedding_model=C.EMBEDDING_MODEL, cache=cache, train_file=rb_frame.iloc[small],
                                      models_to_route=list(C.TIERS), **settings["mlp"])
        bodies["mlp"] = lambda i, r=r: r.get_model_to_route_to(ctx.query[i:i + 1])
    if "svm" in settings:
        r = ext.routerbench.SVMRouter(embedding_model=C.EMBEDDING_MODEL, cache=cache, train_file=rb_frame.iloc[small],
                                      models_to_route=list(C.TIERS))
        bodies["svm"] = lambda i, r=r: r.get_model_to_route_to(ctx.query[i:i + 1])
    if "mf" in settings:
        model = sys.modules["llmrouter.models.mfrouter.router"].BilinearMF(dim=128, num_models=3, text_dim=x.shape[1]).eval()
        bodies["mf"] = lambda i, m=model: m.score_all(m.project_text(torch.from_numpy(ctx.query[i])))
    if "hybrid_llm" in settings:
        from sklearn.neural_network import MLPRegressor
        reg = MLPRegressor(hidden_layer_sizes=[128, 64], activation="relu", solver="adam", max_iter=5,
                           random_state=C.SEED).fit(x[small], y[small, 0])
        bodies["hybrid_llm"] = lambda i, r=reg: np.clip(r.predict(ctx.query[i:i + 1]), 0, 1)
    if "irt_router" in settings:
        from sklearn.neighbors import NearestNeighbors
        cdm = ext.irt.MIRT(ctx.llm_vectors.shape[1], x.shape[1], 25)
        nn_ = NearestNeighbors(n_neighbors=5).fit(x)
        lam = settings["irt_router"]["lamda"]
        llm = torch.from_numpy(ctx.llm_vectors)

        def irt(i, cdm=cdm):
            q = ctx.query[i:i + 1]
            if lam > 0:
                q = (1 - lam) * q + lam * x[nn_.kneighbors(q)[1][0]].mean(axis=0, keepdims=True)
            with torch.no_grad():
                return cdm.generate(llm, torch.from_numpy(np.repeat(q.astype(np.float32), 3, axis=0)))
        bodies["irt_router"] = irt
    if "avengers_pro" in settings:
        s = settings["avengers_pro"]
        config = ext.avengers.config.SimpleClusterConfig(
            data_path=str(ctx.inputs_path), seed=C.SEED, n_clusters=s["n_clusters"], top_k=3, beta=9.0, max_router=1,
            performance_weight=s["performance_weight"], cost_sensitivity=round(1 - s["performance_weight"], 10),
            embedding_model=C.EMBEDDING_MODEL)
        router = ext.avengers.balance.BalanceClusterRouter(config)
        router.available_models = list(C.TIERS)
        lookup = dict(zip(frame.sample_id, x))
        router._generate_embeddings_concurrent = lambda queries: [lookup[q] for q in queries]
        cost = ctx.inputs["service_cost"]
        router.build_cluster_model([{"query": sid, "records": {t: float(y[k, j]) for j, t in enumerate(C.TIERS)},
                                     "usages": {t: {"cost": cost[t]} for t in C.TIERS}}
                                    for k, sid in enumerate(frame.sample_id)])
        bodies["avengers_pro"] = lambda i, r=router: C.avengers_scores(r, ctx.query[i:i + 1], list(C.TIERS))
    if "r1_bge" in settings:
        h = settings["r1_bge"]["hidden"]
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], h), torch.nn.GELU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(h, 3)).eval()
        bodies["r1_bge"] = lambda i, n=net: torch.sigmoid(n(torch.from_numpy(ctx.query[i:i + 1])))
    if "graphrouter" in settings:
        bodies["graphrouter"] = graph_body(ctx, ext, settings["graphrouter"], frame, x, y)
    return bodies


def graph_body(ctx, ext, setting: dict, frame, x, y):
    """One new question added to the training graph and scored, as GraphRouter's route_single does."""
    import torch
    from sklearn.preprocessing import MinMaxScaler
    if setting.get("impl") == "official":
        gr = ext.graphrouter
        graphs = {}
        for source in C.SOURCES:
            pool = ctx.inputs["members"][source]
            names = [m["model"] for m in pool]
            out = ctx.inputs["model_outcomes"]
            wide = out[out.source_dataset.eq(source)].pivot(index="sample_id", columns="model", values="success")[names]
            q_train = ctx.inputs["vectors"][ctx.inputs["row"].loc[wide.index].to_numpy()]
            cost = np.array([m["cost"] for m in pool], float)
            predictor = gr.GNN_prediction(query_feature_dim=q_train.shape[1], llm_feature_dim=1024, hidden_features_size=8,
                                          in_edges_size=3, wandb=C._NoLog(), config={"learning_rate": 3e-4, "weight_decay": 1e-4,
                                          "model_path": "/tmp/unused", "train_epoch": 1, "batch_size": 1,
                                          "train_mask_rate": 0.5, "llm_num": len(names)}, device="cpu")
            graphs[source] = (gr.form_data("cpu"), predictor, q_train, wide.to_numpy(float).reshape(-1), cost / cost.max(),
                              ctx.model_vectors[source], [m["tier"] for m in pool])
        source_of = ctx.scored.source_dataset.to_numpy()

        def official(i):
            form, predictor, q_train, effect, cost, llm, tiers = graphs[source_of[i]]
            n_train, models = len(q_train), len(llm)
            query = np.vstack([q_train, ctx.query[i:i + 1]])
            n = len(query)
            seen = torch.zeros(n * models)
            seen[: n_train * models] = 1
            new = 1 - seen
            data = C.official_graph(form, query, llm, np.concatenate([effect, np.zeros(models)]), cost,
                                    np.zeros((n * models, 1)), new, seen, torch.zeros(n * models), new)
            predictor.model.eval()
            with torch.no_grad():
                pred = predictor.model(task_id=data.task_id, query_features=data.query_features, llm_features=data.llm_features,
                                       edge_index=data.edge_index, edge_mask=new.bool(), edge_can_see=seen.bool(),
                                       edge_weight=data.combined_edge)
            return C.tier_means(pred.reshape(-1, models).numpy(), tiers)
        return official
    routing = ctx.routing_frame()
    routing["query"] = routing["query"].astype("category")
    ext.llmrouter.MetaRouter.context = {
        "cfg": {"hparam": {"hidden_dim": 64, "learning_rate": 0.001, "weight_decay": 0.0001, "train_epoch": 1, "batch_size": 1,
                           "train_mask_rate": 0.3, "val_split_ratio": 0.2},
                "model_path": {"save_model_path": "/tmp/ev10-latency-graph.pt"},
                "llm_data": {t: {"embedding": ctx.llm_vectors[k].tolist()} for k, t in enumerate(C.TIERS)}},
        "routing_data_train": routing, "query_embedding_data": torch.from_numpy(x)}
    router = ext.llmrouter.graph.GraphRouter(yaml_path=None)
    router.device = "cpu"
    predictor = router.gnn_predictor
    predictor.model.to("cpu").eval()
    router.form_data.device = "cpu"
    scaler = MinMaxScaler().fit(x)

    def llmrouter_route(i):
        new = scaler.transform(ctx.query[i:i + 1])
        n = len(router.query_embedding_list) + 1
        queries = np.vstack([router.query_embedding_list, new])
        seen = torch.zeros(n * 3)
        seen[: (n - 1) * 3] = 1
        mask_new = 1 - seen
        perf = np.concatenate([router.performance_list, np.zeros(3)])
        data = router.form_data.formulation(
            query_feature=queries, llm_feature=router.llm_embedding, org_node=[q for q in range(n) for _ in range(3)],
            des_node=[0, 1, 2] * n, edge_feature=perf, label=np.zeros((n * 3, 1)), edge_mask=mask_new, train_mask=seen,
            valide_mask=torch.zeros(n * 3), test_mask=mask_new)
        with torch.no_grad():
            return predictor.model(query_features=data.query_features, llm_features=data.llm_features,
                                   edge_index=data.edge_index, edge_mask=mask_new.bool(), edge_can_see=seen.bool(),
                                   edge_weight=data.edge_attr)
    return llmrouter_route


# ---------------------------------------------------------------------------------------------
# Routers with their own encoder
# ---------------------------------------------------------------------------------------------

def encoder_routers(run_dir: Path, external_dir: Path, wanted: set, device: str):
    """Yield (method, callable(text) -> scores, note) one router at a time, so only one encoder is in memory."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    def classifier(model_dir, max_length, note):
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

        def run(text):
            enc = tok([text], truncation=True, max_length=max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                return model(**enc).logits
        return run, note
    for name, rel in MODERNBERT.items():
        if name in wanted:
            yield (name, *classifier(run_dir / rel, 512, "trained weights, 512 tokens"))
    if "r0_operational_8192" in wanted:  # the operational R0 is the same ModernBERT classifier, served at 8192 tokens
        yield ("r0_operational_8192", *classifier(run_dir / MODERNBERT["rt08"], 8192,
                                                   "RT-08 weights (same architecture), 8192 tokens"))
    if "routellm_released" in wanted:
        from huggingface_hub import snapshot_download
        repo, rev = F.HF_MODELS["released"]
        local = snapshot_download(repo, revision=rev, allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                  "tokenizer_config.json", "sentencepiece.bpe.model", "special_tokens_map.json"])
        yield ("routellm_released", *classifier(local, 512, "released weights"))
    if "routellm_retrained" in wanted:
        repo, rev = F.HF_MODELS["xlmr"]
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev, num_labels=3).to(device).eval()

        def retrained(text):
            enc = tok([text], truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                return model(**enc).logits
        yield ("routellm_retrained", retrained, "same architecture as the retrained model (XLM-R base, 3 classes)")
    if "routerdc" in wanted:
        from transformers import DebertaV2Model, DebertaV2Tokenizer
        C.register_llmrouter(external_dir)
        import importlib
        dcmodel = importlib.import_module("llmrouter.models.routerdc.dcmodel")
        repo, rev = F.HF_MODELS["mdeberta"]
        tok = DebertaV2Tokenizer.from_pretrained(repo, revision=rev, truncation_side="left")
        model = dcmodel.RouterModule(backbone=DebertaV2Model.from_pretrained(repo, revision=rev), hidden_state_dim=768,
                                     node_size=3, similarity_function="cos").to(device).eval()

        def dc(text):
            enc = tok([text], max_length=512, truncation=True, return_tensors="pt")
            with torch.no_grad():
                return model(t=1.0, input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))[0]
        yield ("routerdc", dc, "same architecture as the trained RouterDC (mDeBERTa-v3-base + 3 tier embeddings)")


# ---------------------------------------------------------------------------------------------

def measure(run_dir: Path, lock_path: Path, external_dir: Path, ext_ft_dir: Path, output: Path, device: str,
            per_source: int, only: set | None = None) -> dict:
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else None
    record = json.loads(lock_path.read_text())
    selected = {m: v["selected_setting"] for m, v in record["methods"].items()}
    settings = {}
    keep = (lambda m: only is None or m in only)
    for method, sid in selected.items():
        if not keep(method):
            continue
        if method in C.SETTINGS:
            match = [s for s in C.SETTINGS[method] if C.setting_id(method, s) == sid]
            if match:
                settings[method] = match[0]
    inputs = C.load_inputs(run_dir, smoke=False)
    texts = F.load_inputs(run_dir, smoke=False)
    if list(texts["scored"].sample_id) != list(inputs["scored"].sample_id):
        raise SystemExit("question order differs between the 04 and 05 loaders")
    sample = sample_questions(texts, per_source)
    rows = sample.row.tolist()
    ctx = C.Context(inputs, np.zeros((3, 1024), np.float32), "cpu")
    ctx.inputs_path = run_dir / "ours" / "data" / "blended" / "train.parquet"
    desc = C.group_descriptions(inputs["groups"], inputs["service_cost"])
    ctx.llm_vectors = C.encode_texts([desc[t] for t in C.TIERS], "cpu")
    if inputs["members"] is not None:
        texts_m = C.model_descriptions(inputs["members"])
        ctx.model_vectors = {s: C.encode_texts(texts_m[s], "cpu") for s in C.SOURCES}
    ext = types.SimpleNamespace(routerbench=C.load_routerbench(external_dir), llmrouter=C.load_llmrouter(external_dir),
                                graphrouter=C.load_graphrouter(external_dir), irt=C.load_irt(external_dir),
                                avengers=C.load_avengers(external_dir))
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer(C.EMBEDDING_MODEL, revision=C.EMBEDDING_REVISION, device=device)
    bge.max_seq_length = 512
    decision = DecisionStage()
    results = []
    bge_ms = timed(lambda i: bge.encode([texts["texts"][i]], normalize_embeddings=True), rows, sync)
    decision_ms = timed(lambda i: decision(np.random.default_rng(i).uniform(size=(1, 3))), rows)
    results.append({"router": "bge_encoder", "stage": "encoder", **stats(bge_ms)})
    results.append({"router": "decision_rule", "stage": "decision", **stats(decision_ms)})
    for method, body in bge_bodies(ctx, ext, settings, sample, device).items():
        ms = timed(body, rows)
        body_stats = stats(ms)
        total = stats([b + e + d for b, e, d in zip(ms, bge_ms, decision_ms)])
        added = stats([b + d for b, d in zip(ms, decision_ms)])
        results.append({"router": method, "setting": selected[method], "stage": "body", **body_stats})
        results.append({"router": method, "setting": selected[method], "stage": "total_with_bge", **total})
        results.append({"router": method, "setting": selected[method], "stage": "added_given_bge", **added})
        print(f"[latency] {method}: body {body_stats['median_ms']:.1f} ms, total {total['median_ms']:.1f} ms", flush=True)
    wanted = {m for m in set(selected) | set(MODERNBERT) | {"r0_operational_8192"} if keep(m)}
    import gc
    for method, fn, note in encoder_routers(run_dir, ext_ft_dir, wanted, device):
        ms = timed(lambda i: fn(texts["texts"][i]), rows, sync)
        del fn
        gc.collect()
        total = stats([m + d for m, d in zip(ms, decision_ms)])
        results.append({"router": method, "setting": selected.get(method, "default"), "stage": "total", "note": note, **total})
        print(f"[latency] {method}: total {total['median_ms']:.1f} ms", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results).assign(device=device)
    name = f"latency-{device}.parquet"
    table.to_parquet(output / name, index=False)
    meta = {"device": device, "questions": len(rows), "warmup": WARMUP, "threads": torch.get_num_threads(),
            "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "lock_sha256": C.sha256_file(lock_path), "output_sha256": C.sha256_file(output / name)}
    (output / f"latency-{device}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True, help="<run>/lock/selection-lock.json")
    p.add_argument("--external-dir", type=Path, required=True, help="04 external code (common_track fetch)")
    p.add_argument("--external-finetune-dir", type=Path, required=True, help="05 external code (finetune_track fetch)")
    p.add_argument("--output-dir", type=Path, required=True, help="normally <run>/latency")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--per-source", type=int, default=PER_SOURCE)
    p.add_argument("--only", nargs="+", default=None, help="measure only these routers (testing)")
    a = p.parse_args(argv)
    print(json.dumps(measure(a.run_dir, a.lock, a.external_dir, a.external_finetune_dir, a.output_dir, a.device,
                             a.per_source, set(a.only) if a.only else None), indent=1))


if __name__ == "__main__":
    main()
