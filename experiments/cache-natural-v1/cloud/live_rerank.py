"""운영 FastAPI 리랭커의 실제 지연. 게이트웨이 VM 안에서 돌린다(Spring 과 같은 경로·같은 리전 간 왕복).

읽기 전용 추론 호출만 하고, 호출 사이에 쉬어 운영 부하를 최소화한다. 토큰은 출력하지 않는다.
"""
import json
import random
import statistics as st
import subprocess
import time
import urllib.request

BASE = "http://10.138.0.2:8000"
TOKEN = subprocess.run(["gcloud", "secrets", "versions", "access", "latest",
                        "--secret=llm-router-internal-api-token"], capture_output=True, text=True, check=True).stdout.strip()


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    return (time.perf_counter() - t) * 1000


groups = [json.loads(l) for l in open("/tmp/dataset.jsonl", encoding="utf-8")]
originals = [g["original"] for g in groups]
queries = [g[k] for g in groups for k in ("para_light", "para_heavy", "trap")]
rng = random.Random(7)
long_pool = [" ".join(rng.sample(originals, 7)) for _ in range(50)]


def run(name, fn, n):
    for _ in range(3):
        fn()
    xs = []
    for _ in range(n):
        xs.append(fn())
        time.sleep(0.3)
    xs.sort()
    print(json.dumps({"case": name, "p50_ms": round(st.median(xs), 1), "p95_ms": round(xs[int(0.95 * n) - 1], 1), "n": n}))


run("health_rtt", lambda: call("GET", "/health"), 30)
run("rerank_short_5cand", lambda: call("POST", "/router/rerank",
                                       {"query": rng.choice(queries), "candidates": rng.sample(originals, 5)}), 40)
run("rerank_short_1cand", lambda: call("POST", "/router/rerank",
                                       {"query": rng.choice(queries), "candidates": rng.sample(originals, 1)}), 40)
run("rerank_long_5cand", lambda: call("POST", "/router/rerank",
                                      {"query": rng.choice(long_pool), "candidates": rng.sample(long_pool, 5)}), 20)
run("embed_short", lambda: call("POST", "/router/embed", {"message": rng.choice(queries)}), 30)
