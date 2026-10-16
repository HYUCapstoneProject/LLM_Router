"""Cloud Build 에서 NLI 점수를 계산한다.

MODELS 환경변수: "이름@리비전@최대길이@all|diff" 를 + 로 잇는다.
all 이면 pairs.json 의 모든 행, diff 면 diff_rows.json 의 행만 채점한다. 행 = [a(질문), b(캐시 원질문)].
결과는 버킷 없이 받기 위해 gzip+base64 로 로그에 출력한다.
"""
import base64
import gzip
import json
import os
import time

from sentence_transformers import CrossEncoder

rows = json.load(open("pairs.json", encoding="utf-8"))
diff_rows = json.load(open("diff_rows.json", encoding="utf-8"))
t0 = time.time()
result = {}
for spec in os.environ["MODELS"].split("+"):
    name, rev, max_len, which = spec.split("@")
    ids = list(range(len(rows))) if which == "all" else diff_rows
    t1 = time.time()
    m = CrossEncoder(name, revision=rev, max_length=int(max_len), device="cpu")
    labels = [m.model.config.id2label[i].lower() for i in range(len(m.model.config.id2label))]
    ent = labels.index("entailment")
    con = labels.index("contradiction") if "contradiction" in labels else None
    ab = m.predict([(rows[i][0], rows[i][1]) for i in ids], apply_softmax=True, batch_size=16, show_progress_bar=False)
    ba = m.predict([(rows[i][1], rows[i][0]) for i in ids], apply_softmax=True, batch_size=16, show_progress_bar=False)
    tok = m.tokenizer
    trunc = sum(1 for i in ids if len(tok(rows[i][0], rows[i][1])["input_ids"]) > int(max_len))
    result[name] = {"rev": rev, "max_len": int(max_len), "labels": labels, "rows": ids,
                    "ent_ab": [float(x[ent]) for x in ab], "ent_ba": [float(x[ent]) for x in ba],
                    "con_ab": None if con is None else [float(x[con]) for x in ab],
                    "con_ba": None if con is None else [float(x[con]) for x in ba],
                    "truncated": trunc, "elapsed_s": round(time.time() - t1)}
    print(f"[{time.time() - t0:6.0f}s] {name} rows={len(ids)} labels={labels} truncated={trunc}", flush=True)

blob = base64.b64encode(gzip.compress(json.dumps(result).encode())).decode()
print("===OUT-BEGIN===", flush=True)
for k in range(0, len(blob), 3000):
    print(blob[k:k + 3000], flush=True)
print("===OUT-END===", flush=True)
