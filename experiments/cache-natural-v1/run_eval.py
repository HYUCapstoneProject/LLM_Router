"""cache-natural-v1 평가 실행기. 운영 GCE 게이트웨이 VM 안에서 돌린다.

운영 FastAPI 의 /router/embed, /router/rerank 를 직접 호출해 운영 C2·C3 판정을 재현한다.
Spring 게이트웨이를 거치지 않으므로 캐시·request_log 에 아무것도 쓰지 않고 Anthropic 도 부르지 않는다.

표준 라이브러리만 쓴다(VM 에 numpy 가 없다). 토큰은 Secret Manager 에서 읽고 출력하지 않는다.

    python3 run_eval.py dataset.jsonl refs.json out.jsonl
"""
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = "http://10.138.0.2:8000"
ROW_THRESHOLD = 0.80          # cache_entry.threshold 기본값
CANDIDATE_LIMIT = 5           # llm.cache.rerank.candidate-limit
RERANK_SIM = 0.8774           # llm.cache.rerank.similarity-threshold
RERANK_SCORE = 0.03581769     # llm.cache.rerank.score-threshold


def token():
    out = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest",
         "--secret=llm-router-internal-api-token"],
        check=True, capture_output=True, text=True)
    value = out.stdout.strip()
    if not value:
        raise SystemExit("토큰을 읽지 못했습니다")
    return value


TOKEN = token()


def post(path, body, attempts=4):
    data = json.dumps(body).encode("utf-8")
    for attempt in range(attempts):
        req = urllib.request.Request(
            BASE + path, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + TOKEN,
                     "X-Request-ID": "cache-natural-v1"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts - 1:
                raise
            print(f"retry {path} after {error}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def main(dataset_path, refs_path, out_path):
    groups = [json.loads(l) for l in open(dataset_path, encoding="utf-8") if l.strip()]
    refs = json.load(open(refs_path, encoding="utf-8"))

    texts = []
    for g in groups:
        texts += [g["original"], g["para_light"], g["para_heavy"], g["trap"]]
    texts += refs
    unique = list(dict.fromkeys(texts))

    emb, embed_versions = {}, set()
    t0 = time.time()
    for i, text in enumerate(unique):
        r = post("/router/embed", {"message": text})
        emb[text] = unit(r["embedding"])
        embed_versions.add(r.get("modelVersion"))
        if (i + 1) % 100 == 0:
            print(f"embed {i + 1}/{len(unique)} {time.time() - t0:.0f}s", file=sys.stderr)

    originals = [(g["id"], g["original"]) for g in groups]
    rerank_versions = set()
    out = open(out_path, "w", encoding="utf-8")

    # 오염 점검: 각 원질문과 92문항·58쌍 문장의 최대 코사인
    for gid, text in originals:
        best = max(((dot(emb[text], emb[ref]), ref) for ref in refs), default=(None, None))
        out.write(json.dumps({"record": "contamination", "id": gid,
                              "max_cos": best[0], "nearest_ref": best[1]}) + "\n")

    done = 0
    for g in groups:
        for kind in ("para_light", "para_heavy", "trap"):
            q = g[kind]
            sims = sorted(((dot(emb[q], emb[otext]), oid, otext) for oid, otext in originals),
                          reverse=True)
            top = sims[:CANDIDATE_LIMIT]
            cands = [c for c in top if c[0] >= ROW_THRESHOLD]

            own_text = g["original"]
            own_sim = dot(emb[q], emb[own_text])
            rerank_texts = [c[2] for c in cands]
            if own_text not in rerank_texts:
                rerank_texts.append(own_text)
            r = post("/router/rerank", {"query": q, "candidates": rerank_texts})
            rerank_versions.add(r.get("modelVersion"))
            score_of = dict(zip(rerank_texts, r["scores"]))

            c2 = cands[0][1] if cands else None
            accepted, accepted_score = None, None
            for sim, oid, otext in cands:
                s = score_of[otext]
                if sim >= RERANK_SIM and s >= RERANK_SCORE and (accepted is None or s > accepted_score):
                    accepted, accepted_score = oid, s

            out.write(json.dumps({
                "record": "variant", "id": g["id"], "domain": g["domain"], "kind": kind,
                "trap_type": g["trap_type"] if kind == "trap" else None,
                "own_sim": own_sim, "own_score": score_of[own_text],
                "top1_id": sims[0][1], "top1_sim": sims[0][0],
                "own_rank": next(i for i, s in enumerate(sims) if s[1] == g["id"]) + 1,
                "candidates": [[oid, sim, score_of[otext]] for sim, oid, otext in cands],
                "c2_hit": c2, "c3_hit": accepted,
            }) + "\n")
            done += 1
            if done % 100 == 0:
                print(f"variant {done}/{len(groups) * 3} {time.time() - t0:.0f}s", file=sys.stderr)

    out.write(json.dumps({"record": "meta", "embed_versions": sorted(map(str, embed_versions)),
                          "rerank_versions": sorted(map(str, rerank_versions)),
                          "elapsed_s": round(time.time() - t0, 1),
                          "config": {"row_threshold": ROW_THRESHOLD, "candidate_limit": CANDIDATE_LIMIT,
                                     "rerank_sim": RERANK_SIM, "rerank_score": RERANK_SCORE}}) + "\n")
    out.close()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:4])
