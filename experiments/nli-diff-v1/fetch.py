"""빌드 로그에서 결과를 꺼낸다. builds.txt: "<라벨> <빌드 ID>" 줄.

    python fetch.py
"""
import base64
import gzip
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
for line in open(HERE / "builds.txt", encoding="utf-8"):
    label, bid = line.split()
    status = subprocess.run(["gcloud", "builds", "describe", bid, "--format=value(status)"],
                            capture_output=True, text=True, shell=True).stdout.strip()
    target = HERE / f"results/scores_{label}.json"
    if status != "SUCCESS" or target.exists():
        print(label, status, "(저장됨)" if target.exists() else "")
        continue
    log = subprocess.run(["gcloud", "builds", "log", bid], capture_output=True, text=True, shell=True,
                         encoding="utf-8").stdout
    body = log.split("===OUT-BEGIN===")[1].split("===OUT-END===")[0]
    blob = "".join(l.split()[-1] for l in body.strip().splitlines() if l.strip())
    data = json.loads(gzip.decompress(base64.b64decode(blob)))
    json.dump(data, open(target, "w", encoding="utf-8"))
    print(label, status, {k: (v["truncated"], v["elapsed_s"]) for k, v in data.items()})
