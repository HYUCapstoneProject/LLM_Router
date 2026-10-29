"""Build 06_f6_gain_dataprice.ipynb: 05_f6_gain_metric with tier prices taken from the data.

05 prices every source at Claude's 1:3:5 (0.2/0.6/1.0). This notebook prices each source's three
representative models at their mean cost in the training data (candidate statistics, the same file
EV-10 used for the RouterBench/xRouteBench groups), high = 1. Everything else — predictions, F6
decisions, bootstrap seeds and set order — is copied unchanged from 05, so PRICE_BASIS=claude135
must reproduce 05's saved results exactly. The last check cell enforces that.

Usage: python build_f6_dataprice_notebook.py   (reads 05_f6_gain_metric.ipynb next to this file)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INTRO = """# 06 — 재학습 실험의 첫 시도 이득 $G$를 데이터 가격으로 다시 평가

[05_f6_gain_metric](05_f6_gain_metric.ipynb)는 단계 가격을 Claude 가격 비율 1:3:5(0.2 · 0.6 · 1.0)로 매겼습니다. 이 노트북은 같은 예측으로 **단계 가격만 데이터 속 실제 가격으로** 바꿉니다. 재학습 실험의 단계는 원천마다 대표 모델 하나이므로, 각 대표 모델의 학습 데이터 평균 비용을 high = 1로 맞춰 씁니다. 라우터 비교(EV-10)의 집단 평균 데이터 가격과 같은 원리이고, 가격 출처도 EV-10이 RouterBench · xRouteBench 집단을 만들 때 쓴 후보 통계 파일입니다.

| 원천 | 대표 모델 (low · medium · high) |
|---|---|
| RouterBench | Mistral-7B · Yi-34B · GPT-4 |
| xRouteBench | gpt-oss-20B · gpt-oss-120B · Cogito-671B |

예측, F6의 단계 선택, bootstrap 난수 seed, 평가 데이터 순서는 05와 같습니다. 새 학습이나 LLM 호출은 없고 **Colab CPU 런타임**에서 몇 분이면 끝납니다.

- `PRICE_BASIS = 'data'`(기본): 데이터 가격. 결과는 `MyDrive/LLM_Router/retraining-gain-f6-dataprice-20261022/`에 저장합니다.
- `PRICE_BASIS = 'claude135'`: 05와 같은 1:3:5. 05의 저장 결과와 모든 값이 같아야 합니다(마지막 확인 셀). 계산이 바뀌지 않았는지 볼 때만 씁니다. 결과는 `.../check-claude135/`에 따로 저장합니다.
"""

BASIS_CELL = """# 가격 기준. 'data'(기본) 또는 'claude135'(05 재현 확인용)
PRICE_BASIS = os.environ.get('PRICE_BASIS', 'data')
assert PRICE_BASIS in ('data', 'claude135'), PRICE_BASIS
PRICE_DIR = DRIVE / 'retraining-gain-f6-dataprice-20261022'
OUT_DIR = Path(os.environ.get('F6_GAIN_OUT', PRICE_DIR if PRICE_BASIS == 'data' else PRICE_DIR / 'check-claude135'))
REFERENCE_DIR = DRIVE / 'retraining-gain-f6-20261021'      # 05의 저장 결과(1:3:5)
STATS_PATH = PRICE_DIR / 'inputs/candidate-statistics.parquet'
STATS_SHA = '%STATS_SHA%'
print('가격 기준:', PRICE_BASIS, '| 저장 위치:', OUT_DIR)"""

PRICE_MD = """## 단계 가격

대표 모델은 봉인된 stream의 `model_low · model_medium · model_high`에서 읽고, 원천 안에서 하나로 고정돼 있는지 확인합니다. 가격은 후보 통계의 `mean_cost`(학습 데이터 평균 비용)를 high = 1로 나눈 값입니다. 평가 데이터를 보지 않고 정한 값입니다."""

PRICE_CELL = """rep = full_stream.groupby('source_dataset')[['model_low', 'model_medium', 'model_high']].nunique()
assert (rep == 1).all().all(), f'원천 안에서 대표 모델이 하나가 아닙니다:\\n{rep}'
REP_MODELS = full_stream.groupby('source_dataset')[['model_low', 'model_medium', 'model_high']].first()
if PRICE_BASIS == 'claude135':
    PRICES = {source: PRICE.copy() for source in REP_MODELS.index}
    PRICE_LABEL = 'Claude 가격 비율 1:3:5'
else:
    assert sha256(STATS_PATH) == STATS_SHA, '후보 통계 파일이 고정한 판과 다릅니다'
    INPUT_SHA[str(STATS_PATH.relative_to(DRIVE))] = STATS_SHA
    stats = pd.read_parquet(STATS_PATH)
    PRICES = {}
    for source, models in REP_MODELS.iterrows():
        table = stats[stats.source_dataset.eq(source)].set_index('model_name')['mean_cost']
        cost = table.loc[list(models)].to_numpy(float)
        assert np.all(np.diff(cost) > 0), f'{source}: 대표 모델 가격이 low < medium < high가 아닙니다 {cost}'
        PRICES[source] = cost / cost[-1]
    PRICE_LABEL = '대표 모델의 데이터 평균 가격(high = 1)'
price_table = pd.DataFrame(PRICES, index=list(TIERS)).T
price_table['high/low'] = price_table['high'] / price_table['low']
display(REP_MODELS)
display(price_table)"""

CHECK_MD = """## 05 결과와 같은지 확인 (`PRICE_BASIS = 'claude135'`일 때)

1:3:5로 돌리면 F0~F6의 $G$와 95% 구간, ΔG가 05의 저장 결과와 같아야 합니다. 같으면 바뀐 것은 가격뿐이라는 뜻입니다. 데이터 가격으로 돌릴 때는 건너뜁니다."""

CHECK = """if PRICE_BASIS == 'claude135':
    for name, mine, keys, cols in (('gain-summary.csv', summary, ['set', 'arm'], ['A', 'C', 'H', 'G', 'G_lo', 'G_hi']),
                                   ('delta-vs-f0.csv', deltas, ['set', 'arm'], ['delta_G', 'lo', 'hi'])):
        reference = pd.read_csv(REFERENCE_DIR / name)
        merged = mine.merge(reference, on=keys, suffixes=('', '_05'))
        assert len(merged) == len(mine) == len(reference), f'{name}: 행 수가 05와 다릅니다'
        for col in cols:
            tol = 1e-5 if col in ('G_lo', 'G_hi', 'lo', 'hi') else 1e-12
            gap = (merged[col] - merged[f'{col}_05']).abs().max()
            assert gap < tol, f'{name} {col}이 05와 {gap} 다릅니다'
    print('1:3:5 결과가 05와 일치')
else:
    print('데이터 가격 실행이라 05 비교는 건너뜁니다. 1:3:5 재현은 PRICE_BASIS=claude135로 따로 확인합니다.')"""


def text(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def replace_once(source, old, new):
    assert source.count(old) == 1, f"expected one {old!r}"
    return source.replace(old, new)


def priced_functions(source):
    head, test = source.split("# 두 계산이 같은지 확인", 1)
    first, rest = head.split("\n", 1)
    assert first.startswith("PRICE = np.array([0.2, 0.6, 1.0])")
    rest = rest.replace("PRICE", "price")
    for old, new in (("def baseline(cost, fixed_acc):", "def baseline(cost, fixed_acc, price=PRICE):"),
                     ("def baseline_many(cost, fixed):", "def baseline_many(cost, fixed, price=PRICE):"),
                     ("def gain(choice, s):", "def gain(choice, s, price=PRICE):"),
                     ("h = baseline(cost, s.mean(axis=0))", "h = baseline(cost, s.mean(axis=0), price)"),
                     ("def boot_gain(choice, s, weights, fixed):", "def boot_gain(choice, s, weights, fixed, price=PRICE):"),
                     ("return acc - baseline_many(cost, fixed)", "return acc - baseline_many(cost, fixed, price)")):
        rest = replace_once(rest, old, new)
    # The 1:3:5 self-check stays as it was; add the same check at a data-like price.
    test = test + (
        "\nfor trial in range(300):\n"
        "    price = np.sort(rng.uniform(0.01, 1.0, 3)); price /= price[-1]\n"
        "    fixed_acc = rng.random(3)\n"
        "    cost = rng.uniform(price[0], 1.0)\n"
        "    assert abs(baseline(cost, fixed_acc, price) - baseline_many(np.array([cost]), fixed_acc[None, :], price)[0]) < 1e-9\n"
        "print('임의 가격에서도 행렬판 확인: 300개 사례 일치')\n")
    return first + "\n" + rest + "# 두 계산이 같은지 확인" + test


def main():
    nb = json.loads((HERE / "05_f6_gain_metric.ipynb").read_text())
    cells = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in nb["cells"]]
    assert text(cells[1]).startswith("# 환경 설정")
    assert text(cells[7]).startswith("PRICE = np.array")
    assert "point_rows, window_rows, results" in text(cells[9])
    stats_sha = (HERE / "dataprice_stats.sha256").read_text().split()[0]

    cells[0] = markdown(INTRO)
    setup = text(cells[1])
    setup = replace_once(setup, "OUT_DIR = Path(os.environ.get('F6_GAIN_OUT', DRIVE / 'retraining-gain-f6-20261021'))\n", "")
    setup = replace_once(setup, "REFERENCE_DIR = DRIVE / 'retraining-gain-20261021'\n", "")
    cells[1] = code(setup)
    cells[7] = code(priced_functions(text(cells[7])))
    run = text(cells[9])
    for old, new in (("        s = base.loc[mask, S_COLS].to_numpy(float)\n",
                      "        s = base.loc[mask, S_COLS].to_numpy(float)\n        price = PRICES[source]\n"),
                     ("**gain(ch, s)})", "**gain(ch, s, price)})"),
                     ("boot_gain(ch, s, weights, fixed)", "boot_gain(ch, s, weights, fixed, price)"),
                     ("**gain(ch[w], s[w])})", "**gain(ch[w], s[w], price)})"),
                     ("**gain(oracle_choice(s), s)})", "**gain(oracle_choice(s), s, price)})")):
        run = replace_once(run, old, new)
    cells[9] = code(run)

    save = next(i for i, c in enumerate(cells) if "OUT_DIR.mkdir" in text(c))
    saved = text(cells[save])
    for old, new in (("ml/notebooks/retraining_effect/05_f6_gain_metric.ipynb",
                      "ml/notebooks/retraining_effect/06_f6_gain_dataprice.ipynb"),
                     ("단계 가격 1:3:5, 원천 평균", "단계 가격 {PRICE_LABEL}, 원천 평균"),
                     ("'metric': 'G = A - H(C), first-try success, tier price 0.2/0.6/1.0, upper concave hull of fixed tiers',",
                      "'metric': 'G = A - H(C), first-try success, per-source tier price below, upper concave hull of fixed tiers',\n"
                      "    'price_basis': PRICE_BASIS,\n"
                      "    'prices': {s: [float(x) for x in p] for s, p in PRICES.items()},\n"
                      "    'representative_models': REP_MODELS.to_dict('index'),")):
        saved = replace_once(saved, old, new)
    cells[save] = code(saved)
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and text(c).startswith("## 저장"):
            cells[i] = markdown("## 저장\n\n`OUT_DIR`에 표·그림·요약·manifest를 저장합니다. 데이터 가격이면 "
                                "`MyDrive/LLM_Router/retraining-gain-f6-dataprice-20261022/`, 1:3:5 확인이면 그 아래 "
                                "`check-claude135/`입니다. 05의 결과 폴더는 건드리지 않습니다.")
        if c["cell_type"] == "markdown" and text(c).startswith("## 04 결과와 같은지 확인"):
            cells[i] = markdown(CHECK_MD)
        if c["cell_type"] == "code" and text(c).startswith("reference = pd.read_csv(REFERENCE_DIR / 'gain-summary.csv')"):
            cells[i] = code(CHECK)
    basis_at = 2
    cells[basis_at:basis_at] = [code(BASIS_CELL.replace("%STATS_SHA%", stats_sha))]
    functions_at = next(i for i, c in enumerate(cells) if text(c).startswith("PRICE = np.array"))
    cells[functions_at + 1:functions_at + 1] = [markdown(PRICE_MD), code(PRICE_CELL)]
    nb["cells"] = cells
    (HERE / "06_f6_gain_dataprice.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
