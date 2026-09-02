#!/usr/bin/env python3
"""RouterBench held-out test split에 우리 ModernBERT 라우터를 돌려 예측을 받는다.

`/router/classify` 만 호출하므로 Claude 비용은 0이다. 결과를 CSV로 남겨
`pareto_baselines.py` 가 "우리 라우터" 점을 계산할 수 있게 한다.

전량(3,218건)을 돌리면 팀원의 서빙 서버를 16분간 점유하므로 층화 표본을 쓴다.
--limit 0 을 주면 전량을 돈다.

사용법:
    set ROUTER_API_TOKEN=<토큰>
    python experiments/fetch_router_predictions.py --limit 1000

토큰은 인자로 받지 않는다. 명령 이력에 남기지 않기 위해서다.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

DEFAULT_BASE_URL = "http://34.182.108.1:8000"


def make_connection(base_url: str, timeout: float):
    """연결 하나를 재사용한다. 서빙이 다른 리전이라 수립에만 약 120ms 가 든다."""
    parsed = urlparse(base_url)
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=timeout)
    return http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)


def classify(conn, token: str, message: str):
    body = json.dumps({"message": message})
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Request-ID": f"pareto-{uuid.uuid4()}",
        "Connection": "keep-alive",
    }
    conn.request("POST", "/router/classify", body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    if response.status != 200:
        try:
            code = json.loads(raw).get("code", "UNKNOWN")
        except Exception:
            code = f"HTTP_{response.status}"
        raise RuntimeError(code)
    return json.loads(raw)


def stratified_sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """라벨 비율을 유지한 채 표본을 뽑는다. 표본이 편향되면 정확도 추정이 무너진다."""
    if limit <= 0 or limit >= len(frame):
        return frame
    share = limit / len(frame)
    parts = [
        group.sample(n=max(1, round(len(group) * share)), random_state=seed)
        for _, group in frame.groupby("difficulty_label")
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path,
                        default=Path(".venv-experiments/data/training.parquet"))
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"])
    parser.add_argument("--out", type=Path, default=None,
                        help="기본값: experiments/results/pareto/router_predictions_<split>.csv")
    parser.add_argument("--base-url", default=os.environ.get("ROUTER_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--limit", type=int, default=1000, help="0 이면 전량")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    token = os.environ.get("ROUTER_API_TOKEN")
    if not token:
        sys.exit("ROUTER_API_TOKEN 환경변수가 필요합니다.")

    if args.out is None:
        args.out = Path(f"experiments/results/pareto/router_predictions_{args.split}.csv")

    training = pd.read_parquet(args.training)
    test = training[training.split == args.split]
    sample = stratified_sample(test, args.limit, args.seed).reset_index(drop=True)

    print(f"router   : {args.base_url}")
    print(f"{args.split} split {len(test):,}건 중 {len(sample):,}건 표본")
    print(sample.difficulty_label.value_counts().to_string())
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    conn = make_connection(args.base_url, args.timeout)
    rows, errors = [], 0
    started = time.monotonic()

    for i, record in enumerate(sample.itertuples(index=False), start=1):
        for attempt in (1, 2):
            try:
                result = classify(conn, token, record.model_input_text)
                rows.append({
                    "sample_id": record.sample_id,
                    "true_label": record.difficulty_label,
                    "predicted_tier": result["tier"],
                    "confidence": result.get("confidence"),
                    "model_version": result.get("modelVersion"),
                })
                break
            except Exception as error:
                if attempt == 1:
                    conn.close()
                    conn = make_connection(args.base_url, args.timeout)
                    continue
                errors += 1
                print(f"  {record.sample_id} 실패: {error}")
        if i % 100 == 0:
            rate = i / (time.monotonic() - started)
            print(f"  {i}/{len(sample)}  ({rate:.1f}건/초)")

    conn.close()

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    frame = pd.DataFrame(rows)
    agree = (frame.true_label == frame.predicted_tier).mean()
    print(f"\n성공 {len(rows):,}건 · 실패 {errors}건 · {time.monotonic()-started:.0f}초")
    print(f"라벨 일치율: {agree:.1%}")
    print("\n=== 예측 분포 (참 라벨 -> 예측) ===")
    print(pd.crosstab(frame.true_label, frame.predicted_tier).to_string())
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
