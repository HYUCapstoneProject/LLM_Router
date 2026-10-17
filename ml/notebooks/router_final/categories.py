"""EV-10 question categories: each source's own category and one task type shared by all sources.

Own category: RouterBench domain (`category`), xRouteBench task (`category`), LLMRouterBench benchmark family
(`source_family`), SPROUT source (`source_family`). The shared task type groups them so the pooled data and the
results can be read by kind of question. The grouping is a judgement call (plan 2.4); TASK_TYPE is the only place
it lives and the composition report prints it.

`python categories.py composition ...` (local) writes the category make-up of train/dev/test. Test rows are only
counted; their outcomes (needed tier) are not read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_track as C  # noqa: E402

TASK_TYPE = {  # (source, own category) -> shared task type
    ("routerbench", "math"): "수학", ("routerbench", "code"): "코드", ("routerbench", "knowledge"): "지식 객관식",
    ("routerbench", "commonsense"): "상식·추론", ("routerbench", "conversation"): "열린 대화·지시",
    ("xroutebench", "math"): "수학", ("xroutebench", "gsm8k"): "수학", ("xroutebench", "aime_2020_2024"): "수학",
    ("xroutebench", "mbpp"): "코드", ("xroutebench", "human_eval"): "코드",
    ("xroutebench", "mmlu"): "지식 객관식", ("xroutebench", "mmlu_pro"): "지식 객관식",
    ("xroutebench", "arc_challenge"): "상식·추론", ("xroutebench", "hellaswag"): "상식·추론",
    ("xroutebench", "openbook_qa"): "상식·추론", ("xroutebench", "commonsense_qa"): "상식·추론",
    ("xroutebench", "squad"): "독해", ("xroutebench", "boolq"): "독해",
    ("llmrouterbench", "livemathbench"): "수학", ("llmrouterbench", "aime"): "수학",
    ("llmrouterbench", "livecodebench"): "코드", ("llmrouterbench", "mmlupro"): "지식 객관식",
    ("llmrouterbench", "gpqa"): "전문가 과학·난제", ("llmrouterbench", "hle"): "전문가 과학·난제",
    ("llmrouterbench", "simpleqa"): "사실 회상", ("llmrouterbench", "arenahard"): "열린 대화·지시",
    ("sprout", "openhermes/teknium"): "열린 대화·지시", ("sprout", "Idavidrein/gpqa/gpqa_extended"): "전문가 과학·난제",
}
TYPES = ("수학", "코드", "지식 객관식", "상식·추론", "독해", "전문가 과학·난제", "사실 회상", "열린 대화·지시")


def category_table(inputs: dict, roles=("train", "dev", "test")) -> pd.DataFrame:
    """sample_id -> source, own category and shared task type, from the files 00_prepare read."""
    files = {"train": ["a_train"], "dev": ["a_dev"], "test": ["a_test_routerbench", "a_test_xroutebench"]}
    parts = []
    for role in roles:
        for key in files[role]:
            t = pd.read_parquet(inputs[key], columns=["sample_id", "source_dataset", "category"])
            parts.append(t)
    for source in ("llmrouterbench", "sprout"):
        t = pd.read_parquet(inputs[source], columns=["sample_id", "source_family"]).rename(columns={"source_family": "category"})
        parts.append(t.assign(source_dataset=source))
    table = pd.concat(parts, ignore_index=True).drop_duplicates("sample_id")
    table["task_type"] = [TASK_TYPE.get((s, c)) for s, c in zip(table.source_dataset, table.category)]
    unknown = table[table.task_type.isna()][["source_dataset", "category"]].drop_duplicates()
    if len(unknown):
        raise SystemExit(f"categories without a task type: {unknown.values.tolist()}")
    return table.set_index("sample_id")


def composition(run_dir: Path, repo: Path, data_root: Path, output: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_eval as T
    cfg, inputs = T.resolve_inputs(repo, data_root)
    cats = category_table(inputs)
    import lock as L
    excluded, _ = L.evaluation_exclusions(repo)
    manifest = json.loads((run_dir / "prepare/manifest.json").read_text())
    rows = []
    for role in ("train", "dev", "test"):
        path = run_dir / "prepare" / f"{role}.parquet"
        if C.sha256_file(path) != manifest["outputs"][path.name]["sha256"]:
            raise SystemExit(f"00_prepare output changed: {path.name}")
        cols = ["sample_id", "source_dataset", "excluded_v2", "p_low", "p_medium", "p_high"] + (["needed_tier"] if role != "test" else [])
        f = pd.read_parquet(path, columns=cols)
        f = f[~f.excluded_v2.fillna(False).astype(bool)]
        if role != "train":
            f = f[~f.sample_id.isin(excluded)]
        f = f[f[["p_low", "p_medium", "p_high"]].notna().all(axis=1)]
        f = f.join(cats[["category", "task_type"]], on="sample_id")
        if f.category.isna().any():
            raise SystemExit(f"{role}: questions without a category")
        rows.append(f.assign(role=role).drop(columns=["excluded_v2", "p_low", "p_medium", "p_high"]))
    data = pd.concat(rows, ignore_index=True)
    arena = pd.read_parquet(run_dir / "inputs/routerarena-full.parquet", columns=["Domain"])
    output.parent.mkdir(parents=True, exist_ok=True)
    data.groupby(["role", "source_dataset", "category", "task_type"]).size().rename("rows").reset_index().to_csv(
        output.with_suffix(".csv"), index=False)
    output.write_text(markdown(data, arena))


def markdown(data: pd.DataFrame, arena: pd.DataFrame) -> str:
    L = ["# EV-10 데이터 구성 — 카테고리별 분포", "",
         "00_prepare의 train·dev·test에서 v2 제외 문항, 평가 제외 문항(dev·test의 영어가 아닌 문항과 문서 제공형 SQuAD·BoolQ), "
         "집단 풀 확률이 빈 행을 뺀 "
         "**실제로 쓰는 행**이다. test는 문항 수만 세고 결과(필요 단계)는 읽지 않았다. 표의 숫자는 `categories.py composition`이 만든다.", ""]
    roles = ("train", "dev", "test")
    # 0. needed tier (train, dev only; the test's is an outcome)
    L += ["## 0. 필요 단계 비율 (train·dev)", "",
          "필요 단계 = 집단 풀 확률이 .8 이상인 가장 싼 단계(계획서 2.2). test는 결과라 08에서 연다.", "",
          "| 범위 | train low / medium / high | dev low / medium / high |", "|---|---|---|"]
    def mix(frame):
        v = frame.needed_tier.value_counts(normalize=True).reindex(["low", "medium", "high"]).fillna(0)
        return " / ".join(f"{x:.1%}" for x in v)
    tv = {r: data[data.role.eq(r)] for r in ("train", "dev")}
    L.append("| 전체(행 그대로) | " + " | ".join(mix(tv[r]) for r in ("train", "dev")) + " |")
    eq = []
    for r in ("train", "dev"):
        per = tv[r].groupby("source_dataset").needed_tier.value_counts(normalize=True).unstack().reindex(columns=["low", "medium", "high"]).fillna(0)
        eq.append(" / ".join(f"{x:.1%}" for x in per.mean()))
    L.append("| 전체(원천 같은 비중, 학습이 보는 비율) | " + " | ".join(eq) + " |")
    for src in C.SOURCES:
        L.append(f"| {src} | " + " | ".join(mix(tv[r][tv[r].source_dataset.eq(src)]) for r in ("train", "dev")) + " |")
    L.append("")
    # 1. pooled by task type
    L += ["## 1. 과제 유형별 (원천 합계)", "", "| 과제 유형 | " + " | ".join(f"{r} 행 (비율)" for r in roles) + " | 들어 있는 원천 |",
          "|---|" + "---:|" * len(roles) + "---|"]
    tot = data.groupby("role").size()
    for t in TYPES:
        part = data[data.task_type.eq(t)]
        cells = [f"{(part.role == r).sum():,} ({(part.role == r).sum() / tot[r]:.1%})" for r in roles]
        srcs = ", ".join(f"{s} {n:,}" for s, n in part.groupby("source_dataset").size().sort_values(ascending=False).items())
        L.append(f"| {t} | " + " | ".join(cells) + f" | {srcs} |")
    L.append("| **합계** | " + " | ".join(f"**{tot[r]:,}**" for r in roles) + " | |")
    L += ["", "학습 손실은 원천마다 같은 총 가중치를 주므로(계획서 2.4), 학습에서 실제로 받는 비중은 아래 2절의 원천 안 비율 × 1/4이다.", ""]
    # 2. per source, own category
    L += ["## 2. 원천별 카테고리", ""]
    for s in C.SOURCES:
        part = data[data.source_dataset.eq(s)]
        L += [f"### {s}", "", "| 카테고리 | 과제 유형 | " + " | ".join(roles) + " | 필요 단계 low/medium/high (train) | (dev) |",
              "|---|---|" + "---:|" * len(roles) + "---|---|"]
        st = part.groupby("role").size()
        for (cat, t), g in part.groupby(["category", "task_type"]):
            counts = [f"{(g.role == r).sum():,} ({(g.role == r).sum() / st.get(r, 1):.0%})" for r in roles]
            mix = []
            for r in ("train", "dev"):
                n = g[g.role.eq(r)].needed_tier.value_counts(normalize=True).reindex(["low", "medium", "high"]).fillna(0)
                mix.append(" / ".join(f"{v:.0%}" for v in n) if (g.role == r).any() else "–")
            L.append(f"| {cat} | {t} | " + " | ".join(counts) + f" | {mix[0]} | {mix[1]} |")
        L.append("")
    # 3. RouterArena
    L += ["## 3. RouterArena (진단용, 결과 없음)", "", "| 분야 | 문항 |", "|---|---:|"]
    for d, n in arena.Domain.value_counts().items():
        L.append(f"| {d} | {n:,} |")
    L += ["", "## 4. 과제 유형 대응", "", "원천마다 카테고리 체계가 달라 아래처럼 묶었다(`categories.py` `TASK_TYPE`).", "",
          "| 과제 유형 | 원천 카테고리 |", "|---|---|"]
    for t in TYPES:
        L.append(f"| {t} | " + ", ".join(f"{s}:{c}" for (s, c), v in TASK_TYPE.items() if v == t) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("composition")
    c.add_argument("--run-dir", type=Path, required=True)
    c.add_argument("--repo", type=Path, required=True)
    c.add_argument("--data-root", type=Path, default=None)
    c.add_argument("--output", type=Path, required=True, help="markdown file; a .csv with the counts is written beside it")
    a = p.parse_args(argv)
    composition(a.run_dir, a.repo, a.data_root or a.repo, a.output)


if __name__ == "__main__":
    main()
