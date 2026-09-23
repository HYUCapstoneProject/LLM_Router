#!/usr/bin/env python3
"""리랭커를 실제로 켜면 캐시 오탐이 얼마나 줄고 정탐을 얼마나 잃는지 측정한다.

shadow mode 는 실사용자 트래픽에서 데이터가 쌓이길 기다리는 방식이다. 그런데 이건
실험용 서비스라 트래픽이 없다. 그래서 우리가 직접 데이터를 넣어 검증한다.

방법:
  base 19문항을 캐시에 심었다고 가정하고,
  paraphrase 21(히트해야 함) + hard_negative 37 + filler 15(히트하면 안 됨)를 던져
  두 정책의 히트/미스를 비교한다.

    A) 코사인 단독            — 지금 운영 중인 정책
    B) 코사인 AND 리랭커      — 리랭커를 판정에 쓰면 어떻게 되는가

**Claude 호출도 DB 쓰기도 없다.** 배포된 `/router/embed` 와 `/router/rerank` 만 부른다.
pgvector 의 `1 - (a <=> b)` 가 정규화 벡터의 내적이라는 것은 이미 검증했으므로
(REPORT: L2 norm 1.000000), 유사도 계산을 numpy 로 그대로 재현한다.

    set ROUTER_API_TOKEN=<토큰>
    python experiments/validate_reranker_endtoend.py
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

DEFAULT_BASE_URL = "http://34.182.108.1:8000"


def connect(base_url: str, timeout: float):
    p = urlparse(base_url)
    if p.scheme == "https":
        return http.client.HTTPSConnection(p.hostname, p.port or 443, timeout=timeout)
    return http.client.HTTPConnection(p.hostname, p.port or 80, timeout=timeout)


def post(conn, token: str, path: str, payload: dict) -> dict:
    conn.request("POST", path, body=json.dumps(payload), headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Request-ID": f"rerank-eval-{uuid.uuid4()}",
        "Connection": "keep-alive",
    })
    response = conn.getresponse()
    raw = response.read()
    if response.status != 200:
        raise RuntimeError(f"{path} -> HTTP {response.status}: {raw[:200]!r}")
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path,
                        default=Path("experiments/questions-v2.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/results/rerank-endtoend"))
    parser.add_argument("--base-url", default=os.environ.get("ROUTER_BASE_URL", DEFAULT_BASE_URL))
    # 운영값과 스크리닝 후보값
    parser.add_argument("--cosine-threshold", type=float, default=0.80)
    parser.add_argument("--rerank-cosine-threshold", type=float, default=0.8774)
    parser.add_argument("--rerank-score-threshold", type=float, default=0.03581769)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    token = os.environ.get("ROUTER_API_TOKEN")
    if not token:
        sys.exit("ROUTER_API_TOKEN 환경변수가 필요합니다.")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))["questions"]
    seeds = [q for q in questions if q["kind"] == "base"]
    probes = [q for q in questions if q["kind"] != "base"]
    print(f"캐시에 심을 base {len(seeds)}문항 · 던져볼 probe {len(probes)}문항")

    conn = connect(args.base_url, args.timeout)

    def embed(text: str) -> np.ndarray:
        return np.asarray(post(conn, token, "/router/embed", {"message": text})["embedding"],
                          dtype=np.float64)

    print("임베딩 수집 중...")
    seed_vecs = np.vstack([embed(q["text"]) for q in seeds])
    probe_vecs = np.vstack([embed(q["text"]) for q in probes])
    print(f"  {len(seeds) + len(probes)}건 완료\n")

    rows = []
    for i, probe in enumerate(probes):
        # pgvector 의 1 - (a <=> b). 벡터가 L2 정규화되어 있어 내적과 같다.
        sims = seed_vecs @ probe_vecs[i]
        order = np.argsort(-sims)[: args.top_k]

        # A) 코사인 단독 — 지금 운영 정책
        passed = [j for j in order if sims[j] >= args.cosine_threshold]
        cosine_hit = len(passed) > 0
        cosine_best = int(order[0])

        # B) 코사인 AND 리랭커 — 통과한 후보만 리랭커에 넘긴다
        rerank_hit, rerank_best, best_score = False, None, None
        if passed:
            scores = post(conn, token, "/router/rerank", {
                "query": probe["text"],
                "candidates": [seeds[j]["text"] for j in passed],
            })["scores"]
            for j, score in zip(passed, scores):
                if sims[j] >= args.rerank_cosine_threshold and score >= args.rerank_score_threshold:
                    if best_score is None or score > best_score:
                        rerank_hit, rerank_best, best_score = True, j, score
            if best_score is None:
                best_score = max(scores)

        rows.append({
            "id": probe["id"], "kind": probe["kind"], "negType": probe.get("negType"),
            "text": probe["text"],
            "top_similarity": float(sims[cosine_best]),
            "top_candidate": seeds[cosine_best]["text"],
            "cosine_hit": cosine_hit,
            "rerank_hit": rerank_hit,
            "rerank_score": None if best_score is None else float(best_score),
            "matched": None if rerank_best is None else seeds[rerank_best]["text"],
        })
        if (i + 1) % 20 == 0:
            print(f"  probe {i+1}/{len(probes)}")

    conn.close()

    # paraphrase 는 히트해야 하고, hard_negative 와 filler 는 히트하면 안 된다.
    should_hit = [r for r in rows if r["kind"] == "paraphrase"]
    should_miss = [r for r in rows if r["kind"] in ("hard_negative", "filler")]

    def summarize(key: str) -> dict:
        tp = sum(1 for r in should_hit if r[key])
        fp = sum(1 for r in should_miss if r[key])
        return {
            "recall": tp / len(should_hit),
            "false_hits": fp,
            "false_hit_rate": fp / len(should_miss),
            "tp": tp, "fn": len(should_hit) - tp, "fp": fp,
        }

    a, b = summarize("cosine_hit"), summarize("rerank_hit")
    print(f"\n{'':26}{'recall':>10}{'오탐':>8}{'오탐률':>10}")
    print(f"  A) 코사인 단독 (운영)     {a['recall']:9.1%}{a['false_hits']:8d}{a['false_hit_rate']:9.1%}")
    print(f"  B) 코사인 AND 리랭커     {b['recall']:9.1%}{b['false_hits']:8d}{b['false_hit_rate']:9.1%}")
    print(f"\n  오탐 {a['false_hits']} -> {b['false_hits']} 건"
          f"   정탐 {a['tp']} -> {b['tp']} 건 (전체 {len(should_hit)})")

    blocked = [r for r in should_miss if r["cosine_hit"] and not r["rerank_hit"]]
    lost = [r for r in should_hit if r["cosine_hit"] and not r["rerank_hit"]]
    if blocked:
        print(f"\n=== 리랭커가 막아낸 오탐 {len(blocked)}건 ===")
        for r in blocked:
            print(f"  cos={r['top_similarity']:.4f} score={r['rerank_score']:.6f} "
                  f"[{r['negType'] or r['kind']}]")
            print(f"     질문: {r['text'][:62]}")
            print(f"     후보: {r['top_candidate'][:62]}")
    if lost:
        print(f"\n=== 리랭커 때문에 잃은 정탐 {len(lost)}건 ===")
        for r in lost:
            print(f"  cos={r['top_similarity']:.4f} score={r['rerank_score']:.6f}")
            print(f"     질문: {r['text'][:62]}")
            print(f"     후보: {r['top_candidate'][:62]}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({
        "thresholds": {
            "cosine": args.cosine_threshold,
            "rerank_cosine": args.rerank_cosine_threshold,
            "rerank_score": args.rerank_score_threshold,
        },
        "seeds": len(seeds), "probes": len(probes),
        "cosine_only": a, "cosine_and_rerank": b,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
