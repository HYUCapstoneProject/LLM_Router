"""라벨 검수용 표본을 만든다. 시드 고정. A 는 결과를 숨긴 무작위 표본, B 는 결과에 영향이 큰 표본이다.

    python make_review.py
"""
import json
import random

SEED = 20261010
groups = {}
for line in open("data/dataset.jsonl", encoding="utf-8"):
    g = json.loads(line)
    groups[g["id"]] = g
variants = [json.loads(l) for l in open("results/raw.jsonl", encoding="utf-8")]
variants = [v for v in variants if v["record"] == "variant"]
rng = random.Random(SEED)


def text(v):
    return groups[v["id"]][v["kind"]]


def label(v):
    return "함정" if v["kind"] == "trap" else "같은 뜻"


def note(v):
    g = groups[v["id"]]
    return f"{g['trap_type']}: {g['why']}" if v["kind"] == "trap" else ("가벼운 바꿔쓰기" if v["kind"] == "para_light" else "크게 바꾼 바꿔쓰기")


traps = [v for v in variants if v["kind"] == "trap"]
lights = [v for v in variants if v["kind"] == "para_light"]
heavies = [v for v in variants if v["kind"] == "para_heavy"]
sample_a = rng.sample(traps, 10) + rng.sample(lights, 5) + rng.sample(heavies, 5)
rng.shuffle(sample_a)
chosen = {(v["id"], v["kind"]) for v in sample_a}

passed = [v for v in traps if v["c3_hit"] and (v["id"], v["kind"]) not in chosen]
blocked = sorted([v for v in variants if v["kind"] != "trap" and v["c3_hit"] != v["id"]
                  and (v["id"], v["kind"]) not in chosen], key=lambda v: -v["own_sim"])
sample_b = rng.sample(passed, 5) + blocked[:5]


def row(i, v, show_result):
    g = groups[v["id"]]
    cells = [str(i), v["id"], g["original"], text(v), label(v), note(v)]
    if show_result:
        result = "재사용됨" if v["c3_hit"] else "막힘"
        cells.append(f"{result} (cos {v['own_sim']:.3f}, rr {v['own_score']:.3f})")
    cells.append("")
    return "| " + " | ".join(c.replace("|", "/") for c in cells) + " |"


lines = [
    "# 라벨 검수 표본 (cache-natural-v1)",
    "",
    "판단 기준은 하나입니다. **원질문에 대한 좋은 답을 그대로 돌려줘도 변형 질문에 맞는 답인가?**",
    "",
    "- 맞으면 `같은 뜻`, 틀리거나 부족하면 `함정`이 정답 라벨입니다.",
    "- 마지막 칸에 ✔(라벨 맞음) · ✘(라벨 틀림) · ?(애매)를 적어 주세요. ✘나 ?는 이유를 짧게 덧붙여 주세요.",
    "",
    f"시드 {SEED}. 데이터 해시 `812d176d…`(FREEZE rev 2).",
    "",
    "## A. 무작위 20개 — 라벨 전체의 오류율 확인",
    "",
    "판정 결과는 일부러 숨겼습니다. 함정 10 · 가벼운 바꿔쓰기 5 · 크게 바꾼 바꿔쓰기 5입니다.",
    "",
    "| # | id | 원질문 | 변형 | 라벨 | 메모 | 검수 |",
    "|---|---|---|---|---|---|---|",
]
lines += [row(i + 1, v, False) for i, v in enumerate(sample_a)]
lines += [
    "",
    "## B. 결과에 영향이 큰 10개",
    "",
    "1~5는 운영 C3가 **재사용한 함정**(무작위), 6~10은 C3가 **막은 같은 뜻 질문** 중 코사인이 가장 높은 것입니다.",
    "이 라벨이 틀렸다면 함정 통과율이나 같은 뜻 재사용률이 직접 바뀝니다.",
    "",
    "| # | id | 원질문 | 변형 | 라벨 | 메모 | 판정 | 검수 |",
    "|---|---|---|---|---|---|---|---|",
]
lines += [row(i + 1, v, True) for i, v in enumerate(sample_b)]
lines.append("")
open("review/REVIEW_SAMPLE.md", "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("A", len(sample_a), "B", len(sample_b))
