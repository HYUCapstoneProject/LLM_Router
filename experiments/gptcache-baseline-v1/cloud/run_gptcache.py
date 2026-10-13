"""GPTCache 0.1.44 를 그대로 써서 캐시에 넣고 조회한다 (PLAN.md).

설정마다 기준 0.8(기본) 한 번, 기준 0(점수 수집) 한 번 조회한다. 점수는 hit_callback 으로 받는다.
결과는 버킷 없이 받기 위해 gzip+base64 로 로그에 출력한다.
"""
import base64
import gzip
import json
import shutil
import time

import gptcache
from gptcache import Cache, Config
from gptcache.adapter.api import get, init_similar_cache, put
from gptcache.embedding import Onnx
from gptcache.manager import manager_factory
from gptcache.similarity_evaluation import (OnnxModelEvaluation, SbertCrossencoderEvaluation,
                                            SearchDistanceEvaluation)

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
inputs = json.load(open("inputs.json", encoding="utf-8"))

emb = Onnx()
_orig, _memo = emb.to_embeddings, {}


def memo_embed(data, **kw):
    # 같은 텍스트를 설정마다 다시 임베딩하지 않도록 결과만 기억한다. 값은 Onnx 그대로다.
    if data not in _memo:
        _memo[data] = _orig(data, **kw)
    return _memo[data]


emb.to_embeddings = memo_embed
EVALS = {"G1_distance": SearchDistanceEvaluation, "G2_sbert_quora": SbertCrossencoderEvaluation,
         "G3_onnx_albert_dup": OnnxModelEvaluation}
out = {"meta": {"gptcache": gptcache.__version__, "embedding": emb.model, "dimension": emb.dimension}}

for ds, d in inputs.items():
    shutil.rmtree(f"cache_{ds}", ignore_errors=True)
    # sqlite,faiss 조합에서 퇴출 크기는 max_size 인자가 아니라 eviction_params 로 정해진다(기본 1,000).
    dm = manager_factory("sqlite,faiss", data_dir=f"cache_{ds}", eviction_params={"max_size": 10000},
                         vector_params={"dimension": emb.dimension})
    loader = Cache()
    init_similar_cache(cache_obj=loader, embedding=emb, data_manager=dm)
    put_errors = []
    for cid, text in d["entries"]:
        try:
            put(text, cid, cache_obj=loader)
        except Exception as e:  # noqa: BLE001
            put_errors.append([cid, type(e).__name__, str(e)[:200]])
    log(f"{ds}: put {len(d['entries'])} errors {len(put_errors)} faiss={dm.v.count()}")
    assert dm.v.count() == len(d["entries"]) - len(put_errors), "캐시 항목이 퇴출됐다"
    res = {"put_errors": put_errors, "configs": {}}
    for name, cls in EVALS.items():
        ev = cls()
        r = {}
        for thr in (0.8, 0.0):
            c = Cache()
            init_similar_cache(cache_obj=c, embedding=emb, data_manager=dm, evaluation=ev,
                               config=Config(similarity_threshold=thr))
            ans, errs, t1 = {}, [], time.time()
            for qid, text in d["queries"]:
                cap = []
                try:
                    hit = get(text, cache_obj=c, hit_callback=cap.extend)
                except Exception as e:  # noqa: BLE001
                    errs.append([qid, type(e).__name__, str(e)[:200]])
                    hit = None
                ans[qid] = [hit, float(cap[0][1]) if cap else None]
            r[str(thr)] = {"answers": ans, "errors": errs,
                           "ms_per_query": round((time.time() - t1) * 1000 / len(d["queries"]), 2)}
            log(f"{ds} {name} thr={thr}: hits {sum(1 for v in ans.values() if v[0])} errors {len(errs)} "
                f"{r[str(thr)]['ms_per_query']}ms/q")
        res["configs"][name] = r
    out[ds] = res

blob = base64.b64encode(gzip.compress(json.dumps(out).encode())).decode()
print("===OUT-BEGIN===", flush=True)
for k in range(0, len(blob), 3000):
    print(blob[k:k + 3000], flush=True)
print("===OUT-END===", flush=True)
