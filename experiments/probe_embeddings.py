#!/usr/bin/env python3
"""
질문셋을 /router/embed 로 임베딩해 캐시가 실제로 동작할지 검증한다.

Claude 를 호출하지 않고 DB 도 건드리지 않는다. 비용 0.

확인하는 것:
  1. 계약 검증 — 1024차원인가, L2 정규화되어 있는가
     정규화가 안 돼 있으면 pgvector 의 `유사도 = 1 - 거리` 가 성립하지 않아
     threshold 0.80 이 전혀 다른 의미가 된다.
  2. paraphrase 가 잡히는가 (true hit)
  3. hard_negative 가 안 잡히는가 (false hit)
  4. threshold 를 얼마로 두면 둘이 갈리는가

사용법:
    python experiments/probe_embeddings.py
"""

import argparse
import http.client
import json
import math
import os
import sys
import uuid
from itertools import combinations
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://34.182.108.1:8000"
DEFAULT_QUESTIONS = os.path.join(os.path.dirname(__file__), "questions-v1.json")


def make_connection(base_url, timeout):
    parsed = urlparse(base_url)
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=timeout)
    return http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)


def embed(conn, token, message):
    body = json.dumps({"message": message})
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Request-ID": f"probe-{uuid.uuid4()}",
        "Connection": "keep-alive",
    }
    conn.request("POST", "/router/embed", body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    if response.status != 200:
        try:
            raise RuntimeError(json.loads(raw).get("code", f"HTTP_{response.status}"))
        except json.JSONDecodeError:
            raise RuntimeError(f"HTTP_{response.status}")
    return json.loads(raw)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("ROUTER_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--csv", help="설계된 쌍을 CSV 로 내보낸다 (리랭커 평가용)")
    args = parser.parse_args()

    token = os.environ.get("ROUTER_API_TOKEN")
    if not token:
        sys.exit("ROUTER_API_TOKEN 환경변수가 필요합니다.")

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    print(f"router   : {args.base_url}")
    print(f"questions: {len(questions)}개\n")

    conn = make_connection(args.base_url, args.timeout)
    vectors, model_version = {}, None

    for q in questions:
        try:
            result = embed(conn, token, q["text"])
        except Exception:
            conn.close()
            conn = make_connection(args.base_url, args.timeout)
            result = embed(conn, token, q["text"])
        vectors[q["id"]] = result["embedding"]
        model_version = result.get("modelVersion")
    conn.close()

    by_id = {q["id"]: q for q in questions}

    # ── 1. 계약 검증 ────────────────────────────────────────────────
    print("=" * 64)
    print(f"1. 계약 검증  (modelVersion={model_version})")
    dims = {len(v) for v in vectors.values()}
    norms = [math.sqrt(sum(x * x for x in v)) for v in vectors.values()]
    print(f"   차원      : {dims}  (기대 {{1024}})")
    print(f"   L2 norm   : 최소 {min(norms):.6f} / 최대 {max(norms):.6f}  (기대 1.0)")
    normalized = all(abs(n - 1.0) < 1e-3 for n in norms)
    print(f"   정규화    : {'OK' if normalized else '아님 — threshold 의미가 달라진다'}")

    # ── 2. 설계된 쌍 ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("2. 설계된 쌍  (paraphrase 는 높아야, hard_negative 는 낮아야)")
    groups = {}
    for q in questions:
        if q.get("group"):
            groups.setdefault(q["group"], []).append(q)

    designed = []
    for group, members in sorted(groups.items()):
        base = next((m for m in members if m["kind"] == "base"), None)
        if not base:
            continue
        for m in members:
            if m["kind"] == "base":
                continue
            sim = cosine(vectors[base["id"]], vectors[m["id"]])
            designed.append((m["kind"], sim, base["id"], m["id"]))
            flag = "hit 되어야" if m["kind"] == "paraphrase" else "miss 되어야"
            print(f"   {sim:.4f}  {base['id']:>5} ~ {m['id']:<6} {m['kind']:<14} ({flag})")

    para = [s for k, s, *_ in designed if k == "paraphrase"]
    neg = [s for k, s, *_ in designed if k == "hard_negative"]
    if para and neg:
        print(f"\n   paraphrase    최소 {min(para):.4f}")
        print(f"   hard_negative 최대 {max(neg):.4f}")
        gap = min(para) - max(neg)
        print(f"   분리 여유     {gap:+.4f}  "
              f"{'→ 그 사이에 threshold 를 둘 수 있다' if gap > 0 else '→ 겹친다. 임계값만으로 못 가른다'}")

    # ── 3. hard_negative 유형별 ─────────────────────────────────────
    print("\n" + "=" * 64)
    print("3. hard_negative 유형별  (어떤 방식으로 틀리는가)")
    by_type = {}
    for kind, sim, base_id, other_id in designed:
        if kind != "hard_negative":
            continue
        by_type.setdefault(by_id[other_id].get("negType", "unknown"), []).append((sim, base_id, other_id))
    for neg_type in sorted(by_type, key=lambda t: -max(s for s, *_ in by_type[t])):
        sims = [s for s, *_ in by_type[neg_type]]
        worst = max(by_type[neg_type])
        print(f"   {neg_type:<17} n={len(sims):<2} 최대 {max(sims):.4f} 평균 {sum(sims)/len(sims):.4f}"
              f"   최악: {worst[1]}~{worst[2]}")

    # ── 4. 의도치 않은 고유사도 쌍 ──────────────────────────────────
    print("\n" + "=" * 64)
    print("4. 서로 다른 질문인데 유사도가 높은 쌍  (false hit 위험 상위)")
    unintended = []
    for a, b in combinations(vectors, 2):
        qa, qb = by_id[a], by_id[b]
        same_group_para = (qa.get("group") and qa["group"] == qb.get("group")
                           and {qa["kind"], qb["kind"]} <= {"base", "paraphrase"})
        if same_group_para:
            continue
        unintended.append((cosine(vectors[a], vectors[b]), a, b))
    unintended.sort(reverse=True)
    for sim, a, b in unintended[:10]:
        tag = by_id[b].get("negType") or by_id[a].get("negType") or "cross_group"
        print(f"   {sim:.4f} [{tag:<16}] {a:>6} ~ {b:<7} {by_id[a]['text'][:30]:<32} | {by_id[b]['text'][:28]}")

    # ── 5. threshold 스윕 ───────────────────────────────────────────
    print("\n" + "=" * 64)
    print("5. threshold 스윕")
    print("   thr      recall        false hit    F1")
    best = None
    for thr in [round(0.70 + 0.02 * i, 3) for i in range(16)]:
        tp = sum(1 for s in para if s >= thr)
        fp = sum(1 for s, *_ in unintended if s >= thr)
        recall = tp / len(para) if para else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        if best is None or f1 > best[1]:
            best = (thr, f1)
        mark = "  ← 현재 설정" if abs(thr - 0.80) < 1e-9 else ""
        print(f"   {thr:.3f}   {tp:>3}/{len(para):<3} {recall:>5.0%}   {fp:>4}/{len(unintended):<4}  {f1:.3f}{mark}")

    print(f"\n   recall    = paraphrase {len(para)}쌍 중 잡힌 비율 (높을수록 좋음)")
    print(f"   false hit = 서로 다른 질문 {len(unintended)}쌍 중 잡힌 수 (낮을수록 좋음)")
    print(f"   F1 최대   = threshold {best[0]:.3f} (F1={best[1]:.3f})")

    # ── 6. exact-match 로 걸러지는 비율 ─────────────────────────────
    # 업계 관행은 exact -> semantic 2단 구성이다. exact 단계에서 잡히는 건
    # 임베딩 호출(교차 리전 왕복 120ms) 없이 처리할 수 있다.
    print("\n" + "=" * 64)
    print("6. 참고 — 완전 동일 질문 비율")
    texts = [q["text"].strip().lower() for q in questions]
    dupes = len(texts) - len(set(texts))
    print(f"   질문셋 내 완전 중복: {dupes}건")
    print("   실서비스에서는 동일 질문 재요청이 흔하다. exact-match 층을 앞에 두면")
    print("   그 요청은 embed 호출 없이 처리되어 왕복 지연과 오탐이 모두 0 이 된다.")

    # ── CSV 내보내기 ────────────────────────────────────────────────
    if args.csv:
        import csv as csv_module
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv_module.writer(f)
            w.writerow(["pair_id", "label", "neg_type", "tier",
                        "query", "candidate", "bi_encoder_similarity", "reranker_score"])
            for kind, sim, base_id, other_id in designed:
                other = by_id[other_id]
                w.writerow([
                    f"{base_id}~{other_id}",
                    "positive" if kind == "paraphrase" else "negative",
                    other.get("negType", ""),
                    other["expectedTier"],
                    by_id[base_id]["text"],
                    other["text"],
                    f"{sim:.4f}",
                    "",   # 팀원이 채울 칸
                ])
        print(f"\n   CSV 저장: {args.csv}  ({len(designed)}쌍)")


if __name__ == "__main__":
    main()
