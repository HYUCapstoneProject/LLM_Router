#!/usr/bin/env python3
"""
질문셋을 FastAPI /router/classify 에 직접 던져 실제 티어 분포를 본다.

Claude 를 호출하지 않으므로 비용이 0 이다. 질문셋의 난이도 균형을
맞출 때까지 몇 번이든 돌려도 된다.

사용법:
    set ROUTER_API_TOKEN=<토큰>
    python experiments/probe_router.py

    # 내부 IP 로 (Spring VM 안에서 실행할 때)
    python experiments/probe_router.py --base-url http://10.138.0.2:8000

토큰은 인자로 받지 않는다. 명령 이력에 남기지 않기 위해서다.
    gcloud secrets versions access latest --secret=llm-router-internal-api-token
"""

import argparse
import http.client
import json
import os
import sys
import time
import uuid
from collections import Counter
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://34.182.108.1:8000"
DEFAULT_QUESTIONS = os.path.join(os.path.dirname(__file__), "questions-v1.json")


def make_connection(base_url, timeout):
    """
    연결을 하나 만들어 재사용한다.

    Spring(서울)과 Router(오리건)가 다른 리전이라 연결 수립에만 약 120ms 가 든다.
    매 요청마다 새로 맺으면 문항당 120ms 를 그냥 버린다.
    """
    parsed = urlparse(base_url)
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=timeout)
    return http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)


def classify(conn, token, message):
    body = json.dumps({"message": message})
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Request-ID": f"probe-{uuid.uuid4()}",
        "Connection": "keep-alive",
    }
    started = time.monotonic()
    conn.request("POST", "/router/classify", body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    elapsed_ms = int((time.monotonic() - started) * 1000)

    if response.status != 200:
        # 오류 본문에 토큰은 없지만 그대로 노출하지 않고 code 만 뽑는다
        try:
            code = json.loads(raw).get("code", "UNKNOWN")
        except Exception:
            code = f"HTTP_{response.status}"
        return None, code, elapsed_ms

    return json.loads(raw), None, elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("ROUTER_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    token = os.environ.get("ROUTER_API_TOKEN")
    if not token:
        sys.exit("ROUTER_API_TOKEN 환경변수가 필요합니다. "
                 "gcloud secrets versions access latest --secret=llm-router-internal-api-token")

    with open(args.questions, encoding="utf-8") as f:
        dataset = json.load(f)
    questions = dataset["questions"]

    print(f"router   : {args.base_url}")
    print(f"questions: {len(questions)}개 ({dataset['version']})\n")

    conn = make_connection(args.base_url, args.timeout)
    rows, errors = [], []

    for q in questions:
        try:
            result, err_code, elapsed_ms = classify(conn, token, q["text"])
        except Exception as e:
            # 연결이 끊기면 새로 맺고 한 번만 재시도한다
            conn.close()
            conn = make_connection(args.base_url, args.timeout)
            try:
                result, err_code, elapsed_ms = classify(conn, token, q["text"])
            except Exception:
                errors.append((q["id"], type(e).__name__))
                print(f"  {q['id']:<6} ERROR {type(e).__name__}")
                continue

        if err_code:
            errors.append((q["id"], err_code))
            print(f"  {q['id']:<6} ERROR {err_code}")
            continue

        actual = result["tier"]
        expected = q["expectedTier"]
        mark = "  " if actual == expected else "!!"
        rows.append({**q, "actual": actual, "confidence": result.get("confidence"), "ms": elapsed_ms})
        print(f"{mark} {q['id']:<6} {expected:<6} -> {actual:<6} "
              f"conf={result.get('confidence', 0):.2f} {elapsed_ms:>4}ms  {q['text'][:52]}")

    conn.close()

    if not rows:
        sys.exit("\n성공한 요청이 없습니다.")

    print("\n" + "=" * 60)

    actual_dist = Counter(r["actual"] for r in rows)
    expected_dist = Counter(r["expectedTier"] for r in rows)
    print("티어 분포 (의도 -> 실제)")
    for tier in ("easy", "medium", "hard"):
        print(f"  {tier:<7} {expected_dist[tier]:>3} -> {actual_dist[tier]:>3}")

    agreed = sum(1 for r in rows if r["actual"] == r["expectedTier"])
    print(f"\n의도와 일치: {agreed}/{len(rows)} ({agreed / len(rows):.0%})")

    mismatches = [r for r in rows if r["actual"] != r["expectedTier"]]
    if mismatches:
        print("\n불일치 — 질문을 바꾸거나 의도 라벨을 고친다")
        for r in mismatches:
            print(f"  {r['id']:<6} {r['expectedTier']} -> {r['actual']}  {r['text'][:60]}")

    # 같은 group 의 base 와 paraphrase 가 다른 티어면 캐시 히트 시연이 불안정해진다
    groups = {}
    for r in rows:
        if r.get("group"):
            groups.setdefault(r["group"], []).append(r)
    split = {g: rs for g, rs in groups.items()
             if len({x["actual"] for x in rs if x["kind"] in ("base", "paraphrase")}) > 1}
    if split:
        print("\n같은 group 인데 티어가 갈림 — 캐시 임계값이 티어별로 다르면 문제가 된다")
        for g, rs in split.items():
            detail = ", ".join(f"{x['id']}={x['actual']}" for x in rs)
            print(f"  {g}: {detail}")

    if errors:
        print(f"\n실패 {len(errors)}건: {errors}")

    latencies = sorted(r["ms"] for r in rows)
    print(f"\nclassify 지연: 중앙값 {latencies[len(latencies) // 2]}ms, 최대 {latencies[-1]}ms")


if __name__ == "__main__":
    main()
