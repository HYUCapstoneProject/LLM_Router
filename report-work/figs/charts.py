"""Report charts as standalone SVG files (rendered to PNG by render.sh).

All numbers are copied from the source reports; nothing is recomputed except
simple arithmetic for drawing positions.
"""
import html
import os

OUT = os.path.dirname(os.path.abspath(__file__))
FONT = "Apple SD Gothic Neo, Noto Sans KR, sans-serif"
INK, MUTE, GRID, AXIS = "#24262b", "#666b75", "#e3e5e9", "#9aa0aa"
GRAY = "#b9bdc5"
CACHE, ROUTER, OPS = "#127a73", "#4a46b8", "#b4621a"
GOOD, BAD = "#2f8f4e", "#c2412d"


def esc(s):
    return html.escape(str(s))


def text(x, y, s, size=13, fill=INK, anchor="start", weight=400, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{esc(s)}</text>')


def svg(w, h, body, title, subtitle=None):
    head = text(24, 34, title, 17, INK, weight=700)
    if subtitle:
        head += text(24, 56, subtitle, 12.5, MUTE)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="#fff"/>{head}{body}</svg>')


def save(name, content):
    with open(os.path.join(OUT, name + ".svg"), "w") as f:
        f.write(content)


# ---------------------------------------------------------------- C1 cache judge
def chart_cache_judge():
    # (method, reuse of same-meaning %, trap pass %), page 15 table 1
    sets = [
        ("짧은 질문 (대화형)", [("코사인만", 92.4, 68.8), ("코사인 + 리랭커 (기존 운영)", 49.5, 12.5),
                           ("코사인 + NLI (교체 후)", 84.1, 4.7), ("GPTCache 기본", 90.6, 82.8),
                           ("GPTCache + 리랭커", 38.5, 8.9)]),
        ("긴 문제 (D2 재현본)", [("코사인만", 100, 99.5), ("코사인 + 리랭커 (기존 운영)", 90.0, 69.0),
                            ("코사인 + NLI (교체 후)", 95.5, 1.0), ("GPTCache 기본", 99.5, 100),
                            ("GPTCache + 리랭커", 67.5, 29.0)]),
        ("실사용 요청 (새 세트)", [("코사인만", 100, 96.2), ("코사인 + 리랭커 (기존 운영)", 84.0, 59.5),
                              ("코사인 + NLI (교체 후)", 98.0, 4.6), ("GPTCache 기본", 98.0, 100),
                              ("GPTCache + 리랭커", 71.0, 25.2)]),
    ]
    style = {"코사인만": (GRAY, "o"), "코사인 + 리랭커 (기존 운영)": (MUTE, "o"),
             "코사인 + NLI (교체 후)": (CACHE, "o"), "GPTCache 기본": (GRAY, "s"),
             "GPTCache + 리랭커": (MUTE, "s")}
    W, H = 1200, 520
    pw, ph, top, gap = 330, 300, 110, 50
    body = ""
    for i, (name, pts) in enumerate(sets):
        x0 = 70 + i * (pw + gap)
        X = lambda v: x0 + v / 100 * pw
        Y = lambda v: top + ph - v / 100 * ph
        body += f'<rect x="{x0}" y="{top}" width="{pw * 0.25:.1f}" height="{ph * 0.35:.1f}" fill="{CACHE}" fill-opacity="0.07"/>'
        body += text(x0 + 6, top + ph * 0.35 - 8, "좋은 영역", 11.5, CACHE)
        for t in range(0, 101, 25):
            body += f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{top + ph}" stroke="{GRID}"/>'
            body += f'<line x1="{x0}" x2="{x0 + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}"/>'
            body += text(X(t), top + ph + 18, f"{t}%", 11, MUTE, "middle")
            if i == 0:
                body += text(x0 - 8, Y(t) + 4, f"{t}%", 11, MUTE, "end")
        body += f'<rect x="{x0}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="{AXIS}"/>'
        body += text(x0 + pw / 2, top - 12, name, 14, INK, "middle", 700)
        body += text(x0 + pw / 2, top + ph + 40, "함정 질문 통과율 (낮을수록 안전)", 12, MUTE, "middle")
        for m, reuse, trap in pts:
            c, shape = style[m]
            cx, cy = X(trap), Y(reuse)
            r = 8 if "NLI" in m else 6.5
            if shape == "o":
                body += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{c}" stroke="#fff" stroke-width="1.5"/>'
            else:
                body += f'<rect x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" fill="{c}" stroke="#fff" stroke-width="1.5"/>'
            if "NLI" in m:
                body += text(cx + 14, cy + 26, f"NLI {trap}%", 12.5, CACHE, weight=700)
            if "기존 운영" in m:
                body += text(cx + 10, cy + 18, f"기존 {trap}%", 12, MUTE)
    body += f'<text x="34" y="{top + ph / 2}" font-size="12" fill="{MUTE}" text-anchor="middle" transform="rotate(-90 34 {top + ph / 2})">같은 뜻 질문 재사용률 (높을수록 절감)</text>'
    ly = top + ph + 74
    lx = 70
    for m in ["코사인 + NLI (교체 후)", "코사인 + 리랭커 (기존 운영)", "코사인만", "GPTCache 기본", "GPTCache + 리랭커"]:
        c, shape = style[m]
        if shape == "o":
            body += f'<circle cx="{lx + 6}" cy="{ly - 4}" r="6" fill="{c}"/>'
        else:
            body += f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" fill="{c}"/>'
        body += text(lx + 18, ly, m, 12.5, INK)
        lx += 18 + len(m) * 10.5 + 26
    save("c1-cache-judge", svg(W, H, body, "NLI 판정은 같은 뜻 재사용을 유지하면서 함정 통과를 5% 이하로 줄였다",
                               "세 시험 세트에서 판정 방식별 같은 뜻 재사용률과 함정 통과율 (test, 15번 실험)"))


# ---------------------------------------------------------------- C2 cache D1
def chart_cache_d1():
    ratios = [0, 10, 30, 50]
    save_ = {"C1 exact": [0.0, 10.0, 30.0, 50.0], "C2 의미 캐시": [18.4, 26.1, 41.6, 57.4],
             "C3 의미 캐시 + 리랭커": [3.3, 12.9, 32.0, 51.4]}
    wrong = {"C1 exact": [None, 0, 0, 0], "C2 의미 캐시": [100, 67.0, 36.6, 22.7],
             "C3 의미 캐시 + 리랭커": [100, 24.1, 8.3, 4.7]}
    col = {"C1 exact": CACHE, "C2 의미 캐시": GRAY, "C3 의미 캐시 + 리랭커": MUTE}
    W, H = 1100, 470
    pw, ph, top = 420, 280, 110
    body = ""
    for p, (title, data, ymax) in enumerate([("LLM 호출 절감 (높을수록 좋음)", save_, 60),
                                              ("재사용한 답 중 다른 질문의 답 (낮을수록 좋음)", wrong, 100)]):
        x0 = 80 + p * (pw + 110)
        X = lambda r: x0 + r / 50 * pw
        Y = lambda v: top + ph - v / ymax * ph
        body += text(x0, top - 16, title, 14, INK, weight=700)
        for t in range(0, ymax + 1, 20):
            body += f'<line x1="{x0}" x2="{x0 + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}"/>'
            body += text(x0 - 8, Y(t) + 4, f"{t}%", 11, MUTE, "end")
        for r in ratios:
            body += text(X(r), top + ph + 18, f"{r}%", 11.5, MUTE, "middle")
        body += text(x0 + pw / 2, top + ph + 42, "요청 중 반복 질문 비율", 12, MUTE, "middle")
        body += f'<line x1="{x0}" x2="{x0 + pw}" y1="{top + ph}" y2="{top + ph}" stroke="{AXIS}"/>'
        for k, vals in data.items():
            pts = [(X(r), Y(v)) for r, v in zip(ratios, vals) if v is not None]
            d = "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts)
            sw = 3 if k == "C1 exact" else 2.2
            body += f'<path d="{d}" fill="none" stroke="{col[k]}" stroke-width="{sw}"/>'
            for a, b in pts:
                body += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4.5" fill="{col[k]}"/>'
    lx = 80
    for k in ["C1 exact", "C3 의미 캐시 + 리랭커", "C2 의미 캐시"]:
        body += f'<line x1="{lx}" x2="{lx + 26}" y1="{top + ph + 70}" y2="{top + ph + 70}" stroke="{col[k]}" stroke-width="3"/>'
        body += text(lx + 34, top + ph + 75, {"C1 exact": "exact 캐시만 (C1)", "C3 의미 캐시 + 리랭커": "의미 캐시 + 리랭커 (C3, 기존 운영)", "C2 의미 캐시": "의미 캐시만 (C2)"}[k], 12.5, INK, weight=700 if k == "C1 exact" else 400)
        lx += 34 + len(k) * 11 + 110
    save("c2-cache-d1", svg(W, H + 20, "".join(body),
                            "exact 캐시는 반복된 만큼 정확히 아꼈고, 의미 캐시가 더 아낀 몫은 다른 질문의 답이었다",
                            "반복 요청 재생 시험 4,000건, seed 3개 평균 · 캐시 방식 비교 실험(5.1절)"))


# ---------------------------------------------------------------- C3 12 dumbbell
def chart_unified():
    rows = [("R0 (운영)", 59.4, 87.2), ("R2", 74.7, 86.4), ("R3", 76.7, 85.4), ("R1", 79.4, 84.2),
            ("F3", 74.9, 83.8), ("F1", 79.5, 83.3), ("F2", 79.0, 83.2)]
    W, H = 1000, 470
    x0, x1, top, rh = 170, 900, 110, 42
    lo, hi = 55, 92
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    body = ""
    for t in range(55, 93, 5):
        body += f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 10}" y2="{top + rh * len(rows)}" stroke="{GRID}"/>'
        body += text(X(t), top + rh * len(rows) + 18, f"{t}%", 11.5, MUTE, "middle")
    for i, (m, first, final) in enumerate(rows):
        y = top + i * rh + rh / 2
        hl = m.startswith("R0")
        c = ROUTER if hl else MUTE
        body += text(x0 - 16, y + 5, m, 13.5, INK, "end", 700 if hl else 400)
        body += f'<line x1="{X(first):.1f}" x2="{X(final):.1f}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="{3 if hl else 2}" stroke-opacity="0.55"/>'
        body += f'<circle cx="{X(first):.1f}" cy="{y}" r="6.5" fill="#fff" stroke="{c}" stroke-width="2"/>'
        body += f'<circle cx="{X(final):.1f}" cy="{y}" r="7" fill="{c}"/>'
        body += text(X(first) - 12, y + 5, f"{first}", 12, MUTE, "end")
        body += text(X(final) + 12, y + 5, f"{final}", 12.5, c, weight=700)
    ly = top + rh * len(rows) + 50
    body += f'<circle cx="{x0}" cy="{ly - 4}" r="6" fill="#fff" stroke="{MUTE}" stroke-width="2"/>' + text(x0 + 12, ly, "첫 시도 해결", 12.5)
    body += f'<circle cx="{x0 + 130}" cy="{ly - 4}" r="6" fill="{MUTE}"/>' + text(x0 + 142, ly, "최종 해결 (틀리면 상향 포함)", 12.5)
    body += text(x0, ly + 26, "R0~R3: 학습 목표가 다른 라우터(R0 = 난이도 분류, R1 = 모델별 성공 확률, R2 = 모델 간 우열, R3 = 비용까지 고려한 최선 단계) · F1~F3: 피드백으로 재학습한 R0(5.4절)", 12, MUTE)
    save("c3-unified", svg(W, H + 30, body, "처음부터 높게 보내는 방식은 첫 시도는 높지만 최종 해결률은 R0가 가장 높다",
                           "같은 4,434문항에서 방식별 첫 시도 해결률과 최종 해결률 · 같은 문항 재비교(5.2절)"))


# ---------------------------------------------------------------- C4 EV-04 ceiling
def chart_ceiling():
    rows = [("oracle (각 단계 결과를 미리 앎)", .199), ("다른 LLM 8개의 정답 여부 + 과제", .052),
            ("R1 (학습한 라우터)", .036), ("정답 난이도 라벨", .036), ("과제 이름만", .033), ("R0 점수 + 문턱만 다시 맞춤", .020)]
    W, H = 1000, 380
    x0, x1, top, rh = 300, 900, 96, 40
    X = lambda v: x0 + v / .2 * (x1 - x0)
    body = ""
    for t in [0, .05, .1, .15, .2]:
        body += f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 6}" y2="{top + rh * len(rows)}" stroke="{GRID}"/>'
        body += text(X(t), top + rh * len(rows) + 18, f"+{t:.2f}", 11.5, MUTE, "middle")
    wall = X(.052)
    body += f'<rect x="{x0}" y="{top - 6}" width="{wall - x0:.1f}" height="{rh * len(rows) + 6}" fill="{ROUTER}" fill-opacity="0.06"/>'
    body += f'<line x1="{wall:.1f}" x2="{wall:.1f}" y1="{top - 6}" y2="{top + rh * len(rows)}" stroke="{ROUTER}" stroke-dasharray="5 4"/>'
    body += text(wall + 6, top - 12, "질문 외 정보(다른 LLM 8개의 정답)까지 줘도 +.052", 12, ROUTER)
    for i, (m, v) in enumerate(rows):
        y = top + i * rh
        c = GRAY if i == 0 else (ROUTER if m.startswith("정답") or m.startswith("R1") else MUTE)
        body += text(x0 - 12, y + rh / 2 + 5, m, 13, INK, "end", 700 if m.startswith("정답") else 400)
        body += f'<rect x="{x0}" y="{y + 8}" width="{X(v) - x0:.1f}" height="{rh - 16}" fill="{c}" rx="3"/>'
        body += text(X(v) + 8, y + rh / 2 + 5, f"+{v:.3f}", 12.5, INK, weight=700 if i else 400)
    body += text(x0, top + rh * len(rows) + 42, "같은 비용에서 \"고정 단계를 섞어 쓰는 선\" 대비 첫 시도 품질의 최대 이득 (RouterBench test)", 12, MUTE)
    save("c4-ceiling", svg(W, H, body, "정답 난이도 라벨을 알려 줘도 라우팅 이득은 +.036에서 멈춘다",
                           "라우팅 천장 분석(5.3.1절) · 봉인 test RouterBench 3,473문항"))


# ---------------------------------------------------------------- C5 EV-04 resolution vs cost
def chart_resolution():
    rows = [("low부터 시작", .292, .901, .483), ("medium부터 시작", .377, .882, .712), ("high만", 1.0, .824, .824),
            ("운영 R0", .352, .885, .572), ("R1", .595, .849, .787), ("oracle", .277, .901, .901)]
    W, H = 1000, 500
    x0, x1, y0, y1 = 110, 900, 110, 410
    X = lambda c: x0 + c / 1.05 * (x1 - x0)
    Y = lambda r: y1 - (r - .80) / .12 * (y1 - y0)
    body = ""
    for t in [0, .2, .4, .6, .8, 1.0]:
        body += f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{y0}" y2="{y1}" stroke="{GRID}"/>'
        body += text(X(t), y1 + 18, f"{t:.1f}", 11.5, MUTE, "middle")
    for t in [.80, .84, .88, .92]:
        body += f'<line x1="{x0}" x2="{x1}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}"/>'
        body += text(x0 - 8, Y(t) + 4, f"{t * 100:.0f}%", 11.5, MUTE, "end")
    body += f'<line x1="{x0}" x2="{x1}" y1="{Y(.901):.1f}" y2="{Y(.901):.1f}" stroke="{ROUTER}" stroke-dasharray="5 4"/>'
    body += text(x1, Y(.901) - 8, "oracle의 최종 해결률 90.1%", 12, ROUTER, "end")
    lab = {"low부터 시작": (-12, 28, "end"), "medium부터 시작": (12, 24, "start"), "high만": (-12, -12, "end"),
           "운영 R0": (12, -10, "start"), "R1": (12, 5, "start"), "oracle": (-12, -12, "end")}
    names = {"운영 R0": "난이도 분류 (배포)", "R1": "후보별 성공 확률 예측"}
    for m, c, r, first in rows:
        hl = m == "low부터 시작"
        col = ROUTER if hl else (GRAY if m == "oracle" else MUTE)
        body += f'<circle cx="{X(c):.1f}" cy="{Y(r):.1f}" r="{8 if hl else 6.5}" fill="{col}" stroke="#fff" stroke-width="1.5"/>'
        dx, dy, a = lab[m]
        body += text(X(c) + dx, Y(r) + dy, f"{names.get(m, m)} (첫 시도 해결률 {first * 100:.0f}%)", 12.5, INK, a, 700 if hl else 400)
    body += text((x0 + x1) / 2, y1 + 44, "총비용 (high만 쓸 때 = 1)", 12, MUTE, "middle")
    body += f'<text x="44" y="{(y0 + y1) / 2}" font-size="12" fill="{MUTE}" text-anchor="middle" transform="rotate(-90 44 {(y0 + y1) / 2})">최종 해결률 (틀리면 상향)</text>'
    save("c5-resolution", svg(W, H, body, "시작 단계별 총비용과 최종 해결률",
                              "시작 정책별 총비용과 최종 해결률 · 라우팅 천장 분석(5.3.1절), RouterBench test 3,473문항"))


# ---------------------------------------------------------------- C6 EV-10 mean gain
def chart_mean_gain():
    rows = [("oracle", .262, .256, .267, "ref"), ("GraphRouter", .044, .039, .048, "p"), ("R1-집단", .043, .038, .047, "o"),
            ("R1-집단 (9월 데이터만)", .041, .036, .045, "o"), ("R0-집단", .037, .033, .041, "o"), ("kNN", .037, .033, .041, "p"),
            ("kNN (9월 데이터만)", .034, .029, .039, "p"), ("SVM", .032, .028, .036, "p"), ("RouterDC", .032, .028, .037, "p"),
            ("IRT-Router", .030, .027, .034, "p"), ("RouteLLM 재학습본", .029, .025, .032, "p"),
            ("RouteLLM 공개본", .024, .021, .028, "p"), ("MF", .017, .015, .020, "p"), ("MLP", .013, .009, .018, "p"),
            ("Avengers-Pro", .013, .010, .016, "p"), ("Hybrid LLM", .009, .004, .013, "p")]
    W, H = 1000, 852
    x0, x1, top, rh = 210, 760, 110, 38
    X = lambda v: x0 + v / .05 * (x1 - x0)
    body = ""
    for t in [0, .01, .02, .03, .04, .05]:
        body += f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 6}" y2="{top + rh * len(rows)}" stroke="{GRID}"/>'
        body += text(X(t), top + rh * len(rows) + 18, f"+{t:.2f}", 11.5, MUTE, "middle")
    for i, (m, v, lo, hi, kind) in enumerate(rows):
        y = top + i * rh + rh / 2
        c = ROUTER if kind == "o" else (GRAY if kind == "ref" else MUTE)
        body += text(x0 - 14, y + 5, m, 13, INK, "end", 700 if kind == "o" else 400)
        if kind == "ref":
            body += f'<line x1="{x0}" x2="{x1 + 30}" y1="{y}" y2="{y}" stroke="{GRAY}" stroke-dasharray="3 4"/>'
            body += text(x1 + 40, y + 5, "+.262 (축 밖, 정답을 아는 상한)", 12, MUTE)
            continue
        body += f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2"/>'
        body += f'<circle cx="{X(v):.1f}" cy="{y}" r="6" fill="{c}"/>'
        body += text(X(hi) + 10, y + 5, f"+{v:.3f}", 12, c if kind == "o" else MUTE, weight=700 if kind == "o" else 400)
    body += f'<rect x="{X(.036):.1f}" y="{top + rh - 4}" width="{X(.048) - X(.036):.1f}" height="{rh * 3 + 8}" fill="{ROUTER}" fill-opacity="0.07"/>'
    body += text(x1 + 70, top + rh * 2.6, "상위 3개: 95% 구간이 겹쳐", 12, ROUTER)
    body += text(x1 + 70, top + rh * 2.6 + 17, "서로 구분되지 않음", 12, ROUTER)
    ly = top + rh * len(rows) + 50
    body += f'<circle cx="{x0}" cy="{ly - 4}" r="6" fill="{ROUTER}"/>' + text(x0 + 12, ly, "우리 라우터", 12.5)
    body += f'<circle cx="{x0 + 120}" cy="{ly - 4}" r="6" fill="{MUTE}"/>' + text(x0 + 132, ly, "논문 라우터", 12.5)
    body += text(x0 + 250, ly, "선 = 95% 구간 (질문 묶음 bootstrap 10,000회)", 12.5, MUTE)
    body += text(x0, ly + 24, "-집단: 단계를 원천별 가격대 모델 집단으로 정의해 학습한 라우터", 12, MUTE)
    save("c6-mean-gain", svg(W, H, body, "상위 3개 라우터가 공동 1위지만, 고정 시작 대비 이득은 +.04 수준이다",
                             "원천 4개 평균, 같은 비용의 고정 시작 정책 대비 평균 이득 (κ 21개 × 비용 수준 3개 평균) · 논문 라우터 비교(5.3.3절), test 17,662문항"))


# ---------------------------------------------------------------- C7 EV-10 strata
def chart_strata():
    strata = ["풀리는 문항", "일부만 풀림", "아무도 못 품"]
    rows = [("GraphRouter", [-.009, -.042, -.104], MUTE), ("R1-집단", [-.002, -.063, -.119], ROUTER),
            ("R0-집단", [-.001, -.073, -.192], ROUTER), ("kNN", [-.006, -.055, -.137], MUTE),
            ("oracle", [.145, .152, .000], GRAY)]
    W, H = 1000, 470
    x0, x1, y0, y1 = 110, 940, 100, 380
    lo, hi = -.2, .16
    Y = lambda v: y0 + (hi - v) / (hi - lo) * (y1 - y0)
    gw = (x1 - x0) / 3
    bw = 30
    body = ""
    for t in [-.2, -.15, -.1, -.05, 0, .05, .1, .15]:
        body += f'<line x1="{x0}" x2="{x1}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID if t else AXIS}"/>'
        body += text(x0 - 8, Y(t) + 4, f"{t:+.2f}", 11, MUTE, "end")
    for g, s in enumerate(strata):
        gx = x0 + g * gw + gw / 2 - len(rows) * (bw + 8) / 2
        for k, (m, vals, c) in enumerate(rows):
            v = vals[g]
            bx = gx + k * (bw + 8)
            yA, yB = sorted([Y(0), Y(v)])
            body += f'<rect x="{bx:.1f}" y="{yA:.1f}" width="{bw}" height="{max(yB - yA, 1):.1f}" fill="{c}" fill-opacity="{1 if c != ROUTER or m == "R1-집단" else 0.6}"/>'
        body += text(x0 + g * gw + gw / 2, y1 + 24, s, 13.5, INK, "middle", 700)
    ly = y1 + 56
    lx = x0
    for m, vals, c in rows:
        op = 1 if c != ROUTER or m == "R1-집단" else 0.6
        body += f'<rect x="{lx}" y="{ly - 11}" width="12" height="12" fill="{c}" fill-opacity="{op}"/>' + text(lx + 18, ly, m, 12.5)
        lx += 18 + len(m) * 9 + 30
    body += text(x0 + 6, Y(.145) - 8, "oracle은 풀리는 문항에서도 +.145", 12, MUTE)
    save("c7-strata", svg(W, H, body, "층 안에서는 모든 라우터의 이득이 0 이하다: 이득은 층을 가르는 데서 나온다",
                          "문항을 \"어느 단계든 80% 이상 푸는가\"로 세 층으로 나눈 뒤 층 안에서 다시 잰 평균 이득 · 논문 라우터 비교(5.3.3절)"))


# ---------------------------------------------------------------- C8 latency
def chart_latency():
    rows = [("BGE 기반 (MLP · SVM 등)", 1, "1ms 미만"), ("GraphRouter", 97, "97ms"),
            ("kNN-A", 102, "102ms"), ("kNN", 160, "160ms"), ("ModernBERT 계열 (R0-집단 · R1-집단)", 250, "약 250ms")]
    W, H = 1000, 340
    x0, x1, top, rh = 340, 880, 96, 40
    X = lambda v: x0 + v / 300 * (x1 - x0)
    body = ""
    for t in range(0, 301, 50):
        body += f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 6}" y2="{top + rh * len(rows)}" stroke="{GRID}"/>'
        body += text(X(t), top + rh * len(rows) + 18, f"{t}ms", 11.5, MUTE, "middle")
    for i, (m, v, lab) in enumerate(rows):
        y = top + i * rh
        c = ROUTER if "ModernBERT" in m else MUTE
        body += text(x0 - 12, y + rh / 2 + 5, m, 13, INK, "end")
        body += f'<rect x="{x0}" y="{y + 9}" width="{max(X(v) - x0, 2):.1f}" height="{rh - 18}" fill="{c}" rx="3"/>'
        body += text(X(v) + 8, y + rh / 2 + 5, lab, 12.5, INK, weight=700)
    body += text(x0, top + rh * len(rows) + 42, "서비스가 캐시 때문에 BGE 임베딩을 이미 계산한다고 볼 때 더해지는 지연, Colab CPU 중앙값 (GPU T4에서 ModernBERT는 26~32ms)", 12, MUTE)
    save("c8-latency", svg(W, H, body, "라우팅 판단에 더해지는 지연은 1ms 미만부터 약 250ms까지 갈린다",
                           "검증용 200문항을 한 문항씩, 워밍업 20문항 뒤 측정 · 논문 라우터 비교(5.3.3절)"))


# ---------------------------------------------------------------- C9 retraining F0-F3
def chart_retrain():
    arms = ["F0", "F1", "F2", "F3"]
    panels = [("easy로 보낸 비율", [60.1, 0.0, 0.7, 24.7], "%", 70),
              ("질문당 비용 (×10⁻³ $)", [2.40, 4.24, 4.19, 3.92], "", 5),
              ("최종 해결률", [87.0, 83.1, 83.0, 83.6], "%", 100)]
    W, H = 1100, 420
    pw, top, ph = 290, 120, 210
    body = ""
    for p, (title, vals, unit, ymax) in enumerate(panels):
        x0 = 70 + p * (pw + 70)
        body += text(x0, top - 22, title, 14, INK, weight=700)
        bw = 46
        for i, (a, v) in enumerate(zip(arms, vals)):
            bx = x0 + 20 + i * (bw + 22)
            h = v / ymax * ph
            c = OPS if a == "F0" else GRAY
            body += f'<rect x="{bx}" y="{top + ph - h:.1f}" width="{bw}" height="{max(h, 1.5):.1f}" fill="{c}" rx="3"/>'
            lab = f"{v:.1f}{unit}" if unit else f"{v:.2f}"
            body += text(bx + bw / 2, top + ph - h - 7, lab, 12.5, INK, "middle", 700 if a == "F0" else 400)
            body += text(bx + bw / 2, top + ph + 20, a, 13, INK, "middle", 700 if a == "F0" else 400)
        body += f'<line x1="{x0}" x2="{x0 + pw}" y1="{top + ph}" y2="{top + ph}" stroke="{AXIS}"/>'
    body += text(70, top + ph + 56, "F는 피드백(Feedback) 재학습 비교군 · F0 = 재학습 안 함(운영 1.0) · F1 = 새 피드백만 · F2 = 과거 피드백·초기 데이터 섞기(운영 방식) · F3 = F2 + easy 보충", 12, MUTE)
    save("c9-retrain", svg(W, H, body, "피드백으로 재학습하면 easy가 사라지고 비용은 63~77% 늘며 최종 해결률은 떨어졌다",
                           "피드백 시뮬레이션용 문항 6개 구간, 다음 구간 질문으로 평가 · 피드백 재학습 실험(5.4절)"))


# ---------------------------------------------------------------- C10 F4/F5
def chart_f45():
    W, H = 1100, 440
    body = ""
    # left: label mix
    x0, top, bw, ph = 80, 120, 104, 230
    body += text(x0, top - 24, "학습 데이터의 라벨 비율", 14, INK, weight=700)
    mixes = [("실제 질문", [49, 38, 13]), ("F4 학습 데이터 (오답만)", [27, 49, 24])]
    shades = [(OPS, 0.35), (OPS, 0.7), (OPS, 1.0)]
    for i, (name, parts) in enumerate(mixes):
        bx = x0 + 40 + i * 170
        y = top
        for (c, op), v, lab in zip(shades, parts, ["easy", "medium", "hard"]):
            h = v / 100 * ph
            body += f'<rect x="{bx}" y="{y:.1f}" width="{bw}" height="{h:.1f}" fill="{c}" fill-opacity="{op}"/>'
            body += text(bx + bw / 2, y + h / 2 + 5, f"{lab} {v}%", 11.5, "#fff" if op > 0.5 else INK, "middle", 700)
            y += h
        body += text(bx + bw / 2, top + ph + 22, name, 12.5, INK, "middle")
    # right: accuracy & over-prediction
    x0 = 520
    body += text(x0, top - 24, "봉인 test에서의 난이도 판단", 14, INK, weight=700)
    arms = [("F0 (재학습 안 함)", .619, .113), ("F4 (오답만)", .278, .561), ("F5 (비율 맞춤)", .580, .140)]
    X = lambda v: x0 + 170 + v / .7 * 330
    for t in [0, .2, .4, .6]:
        body += f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{top + 230}" stroke="{GRID}"/>'
        body += text(X(t), top + 248, f"{t:.1f}", 11, MUTE, "middle")
    for i, (a, acc, over) in enumerate(arms):
        y = top + 12 + i * 74
        body += text(x0 + 160, y + 22, a, 13, INK, "end", 700 if a.startswith("F0") else 400)
        body += f'<rect x="{X(0):.1f}" y="{y}" width="{X(acc) - X(0):.1f}" height="18" fill="{OPS}" rx="2"/>'
        body += text(X(acc) + 6, y + 14, f"정확도 {acc:.3f}", 12, INK, weight=700)
        body += f'<rect x="{X(0):.1f}" y="{y + 22}" width="{X(over) - X(0):.1f}" height="18" fill="{GRAY}" rx="2"/>'
        body += text(X(over) + 6, y + 36, f"필요보다 높게 {over:.3f}", 12, MUTE)
    save("c10-f45", svg(W, H, body, "양방향 정답을 줘도 F4는 반대로 쏠렸고, 비율을 맞춘 F5도 재학습 안 한 모델을 넘지 못했다",
                        "양방향 오류 교정 실험(5.4절): F4 = 틀린 문항을 모두 원래 라벨로 재학습, F5 = F4 + 라벨 비율을 실제와 맞춤 · seed 3개 평균, 봉인 test 7,202문항"))


if __name__ == "__main__":
    for f in [chart_cache_judge, chart_cache_d1, chart_unified, chart_ceiling, chart_resolution,
              chart_mean_gain, chart_strata, chart_latency, chart_retrain, chart_f45]:
        f()
    print("ok")
