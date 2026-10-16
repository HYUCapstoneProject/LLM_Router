"""판정 모델 지연 측정 + 작은 NLI 모델 점수 계산 (Cloud Build).

지연은 운영 serving VM(e2-standard-2, vCPU 2)에 맞춰 torch 스레드를 2로 고정해 잰다. 모델마다 별도 프로세스에서
로드해 메모리 증가량도 함께 잰다. 절대값보다 운영 리랭커(quora) 대비 배율을 주로 쓴다.
"""
import json
import os
import random
import statistics as st
import subprocess
import sys
import time

MODELS = [
    "cross-encoder/quora-distilroberta-base",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-xsmall",
    "cross-encoder/nli-distilroberta-base",
    "cross-encoder/nli-MiniLM2-L6-H768",
]


def bench_one(model_name):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import psutil
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import CrossEncoder

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    model = CrossEncoder(model_name, device="cpu")
    load_s = time.perf_counter() - t0
    rss_mb = (proc.memory_info().rss - rss0) / 1e6

    groups = [json.loads(l) for l in open("dataset.jsonl", encoding="utf-8")]
    originals = [g["original"] for g in groups]
    queries = [g[k] for g in groups for k in ("para_light", "para_heavy", "trap")]
    rng = random.Random(7)
    long_pool = [" ".join(rng.sample(originals, 7)) for _ in range(50)]   # 약 350자, D2 원질문 중앙값 368자 수준

    def workloads(short):
        out = []
        for _ in range(60):
            q = rng.choice(queries) if short else rng.choice(long_pool)
            cands = rng.sample(originals, 5) if short else rng.sample(long_pool, 5)
            out.append((q, cands))
        return out

    is_nli = "nli" in model_name
    results = {"model": model_name, "load_s": round(load_s, 2), "rss_mb": round(rss_mb), "threads": torch.get_num_threads()}
    for short in (True, False):
        wl = workloads(short)
        cases = {}
        if is_nli:
            cases["5cand_both_dir"] = [[(q, c) for c in cs] + [(c, q) for c in cs] for q, cs in wl]
            cases["top1_both_dir"] = [[(q, cs[0]), (cs[0], q)] for q, cs in wl]
        else:
            cases["5cand"] = [[(q, c) for c in cs] for q, cs in wl]
            cases["top1"] = [[(q, cs[0])] for q, cs in wl]
        for name, batches in cases.items():
            for b in batches[:5]:
                model.predict(b, show_progress_bar=False)        # warm-up
            times = []
            for b in batches[5:]:
                t = time.perf_counter()
                model.predict(b, show_progress_bar=False)
                times.append((time.perf_counter() - t) * 1000)
            times.sort()
            results[f"{'short' if short else 'long'}_{name}"] = {
                "p50_ms": round(st.median(times), 1), "p95_ms": round(times[int(0.95 * len(times)) - 1], 1), "n": len(times)}
    print("BENCH " + json.dumps(results), flush=True)


def score_small_models():
    """작은 NLI 모델들의 판정 점수(정확도 비교용). 스레드 제한 없이 계산한다."""
    from sentence_transformers import CrossEncoder
    pairs = json.load(open("pairs.json", encoding="utf-8"))
    out = {}
    for name in MODELS[2:]:
        m = CrossEncoder(name, device="cpu")
        labels = [m.model.config.id2label[i].lower() for i in range(len(m.model.config.id2label))]
        ab = m.predict([(p["a"], p["b"]) for p in pairs], apply_softmax=True, batch_size=32, show_progress_bar=False)
        ba = m.predict([(p["b"], p["a"]) for p in pairs], apply_softmax=True, batch_size=32, show_progress_bar=False)
        out[name] = {p["pid"]: {"ab": dict(zip(labels, map(float, x))), "ba": dict(zip(labels, map(float, y)))}
                     for p, x, y in zip(pairs, ab, ba)}
        print(f"scored {name} labels={labels}", flush=True)
    import base64, gzip
    blob = base64.b64encode(gzip.compress(json.dumps(out).encode())).decode()
    print("===SMALL-BEGIN===", flush=True)
    for i in range(0, len(blob), 3000):
        print(blob[i:i + 3000], flush=True)
    print("===SMALL-END===", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        bench_one(sys.argv[1])
    else:
        for name in MODELS:
            subprocess.run([sys.executable, __file__, name], check=True)
        score_small_models()
