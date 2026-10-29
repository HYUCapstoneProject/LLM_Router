"""Architecture diagrams as HTML (rendered to PNG by render.sh).

Facts come from docs/README.md, docs/ml/README.md 5.x, the cache reports and
the SY-03 deployment record. Nothing here is a measured number except where a
report states it.
"""
import html
import os

OUT = os.path.dirname(os.path.abspath(__file__))
COL = {"ui": ("#4b5563", "#f3f4f6"), "spring": ("#2563a8", "#e8f0fa"), "api": ("#127a73", "#e7f4f2"),
       "llm": ("#7a3fb0", "#f3ecfa"), "store": ("#6b7280", "#ffffff"), "ml": ("#b4621a", "#fcf0e3"),
       "router": ("#4a46b8", "#eeedfb"), "note": ("#9aa0aa", "#ffffff"), "hit": ("#2f8f4e", "#eaf6ee")}


def e(s):
    return html.escape(s)


def build(name, W, H, title, sub, nodes, edges, groups=(), notes=()):
    """nodes: id -> (x, y, w, h, kind, title, [lines]); edges: (a, b, side_a, side_b, label, style)."""
    out = [f'<!doctype html><html><head><meta charset="utf-8"><style>'
           f'html,body{{margin:0;background:#fff}}body{{width:{W}px;height:{H}px;position:relative;'
           f'font-family:"Apple SD Gothic Neo","Noto Sans KR",sans-serif;color:#24262b}}'
           f'.n{{position:absolute;box-sizing:border-box;border:2px solid;border-radius:10px;padding:9px 12px}}'
           f'.n b{{display:block;font-size:15.5px;line-height:20px}}.n span{{display:block;font-size:12.5px;line-height:17px;color:#41454d;margin-top:3px}}'
           f'.g{{position:absolute;box-sizing:border-box;border:1.5px dashed #b9bdc5;border-radius:14px}}'
           f'.g i{{position:absolute;left:14px;top:-11px;background:#fff;padding:0 6px;font-style:normal;font-size:13px;font-weight:700;color:#666b75}}'
           f'.lb{{position:absolute;font-size:12px;line-height:15px;color:#41454d;background:#fff;padding:1px 4px;border-radius:4px;white-space:nowrap}}'
           f'.note{{position:absolute;font-size:12.5px;line-height:18px;color:#41454d}}'
           f'</style></head><body data-w="{W}" data-h="{H}">'
           f'<div style="position:absolute;left:24px;top:18px;font-size:19px;font-weight:700">{e(title)}</div>'
           f'<div style="position:absolute;left:24px;top:46px;font-size:13px;color:#666b75">{e(sub)}</div>']
    for gx, gy, gw, gh, gl in groups:
        out.append(f'<div class="g" style="left:{gx}px;top:{gy}px;width:{gw}px;height:{gh}px"><i>{e(gl)}</i></div>')
    for nid, (x, y, w, h, kind, t, lines) in nodes.items():
        b, f = COL[kind]
        dash = "dashed" if kind == "note" else "solid"
        out.append(f'<div class="n" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;border-color:{b};border-style:{dash};background:{f}">'
                   f'<b style="color:{b}">{e(t)}</b>' + "".join(f"<span>{e(l)}</span>" for l in lines) + '</div>')

    def pt(nid, side):
        x, y, w, h = nodes[nid][:4]
        s, _, frac = side.partition(":")
        fr = float(frac) if frac else 0.5
        return {"r": (x + w, y + h * fr), "l": (x, y + h * fr), "t": (x + w * fr, y), "b": (x + w * fr, y + h)}[s]
    svg = [f'<svg style="position:absolute;left:0;top:0" width="{W}" height="{H}"><defs>']
    for k, c in [("k", "#6b7280"), ("g", "#2f8f4e"), ("o", "#b4621a"), ("p", "#7a3fb0"), ("r", "#c2412d")]:
        svg.append(f'<marker id="a{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{c}"/></marker>')
    svg.append('</defs>')
    labels = []
    for a, b, sa, sb, lab, st in edges:
        (x1, y1), (x2, y2) = pt(a, sa), pt(b, sb)
        color = {"k": "#6b7280", "g": "#2f8f4e", "o": "#b4621a", "p": "#7a3fb0", "r": "#c2412d"}[st[0]]
        dash = ' stroke-dasharray="6 5"' if "d" in st else ""
        if sa[0] in "lr" and sb[0] in "lr":
            mx = (x1 + x2) / 2
            d = f"M{x1} {y1} H{mx} V{y2} H{x2}" if abs(y1 - y2) > 1 else f"M{x1} {y1} H{x2}"
            lx, ly = mx, (y1 + y2) / 2 if abs(y1 - y2) > 1 else y1
        elif sa[0] in "tb" and sb[0] in "tb":
            my = (y1 + y2) / 2
            d = f"M{x1} {y1} V{my} H{x2} V{y2}" if abs(x1 - x2) > 1 else f"M{x1} {y1} V{y2}"
            lx, ly = (x1 + x2) / 2, my
        elif sa[0] in "lr":
            d = f"M{x1} {y1} H{x2} V{y2}"
            lx, ly = (x1 + x2) / 2, y1
        else:
            d = f"M{x1} {y1} V{y2} H{x2}"
            lx, ly = x1, (y1 + y2) / 2
        svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash} marker-end="url(#a{st[0]})"/>')
        if lab:
            text, _, pos = lab.partition("@")
            if pos:
                lx, ly = (float(v) for v in pos.split(","))
            labels.append(f'<div class="lb" style="left:{lx}px;top:{ly}px;transform:translate(-50%,-50%);color:{color}">{e(text)}</div>')
    svg.append('</svg>')
    out.append("".join(svg))
    out.extend(labels)
    for nx, ny, nw, ntext in notes:
        out.append(f'<div class="note" style="left:{nx}px;top:{ny}px;width:{nw}px">{ntext}</div>')
    out.append('</body></html>')
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write("".join(out))


# ---------------------------------------------------------------- A1 request + cache path
def a1():
    n = {
        "ui": (30, 110, 150, 70, "ui", "채팅 UI", ["React"]),
        "ex": (240, 110, 200, 90, "spring", "① exact 캐시", ["공백 정리 후 SHA-256", "같은 질문이면 즉시 반환"]),
        "emb": (500, 110, 200, 90, "api", "② BGE 임베딩", ["bge-large-en-v1.5", "질문 → 1,024차원 벡터"]),
        "vec": (760, 110, 220, 90, "spring", "③ pgvector 후보 검색", ["코사인 ≥ 0.80인", "가장 비슷한 후보 1개"]),
        "nli": (1040, 110, 230, 108, "api", "④ NLI 재사용 판정", ["두 질문의 다른 문장만 비교", "양방향 함의 점수의 낮은 쪽", "≥ 0.2815이면 재사용"]),
        "hit": (1040, 300, 230, 70, "hit", "캐시 답 반환", ["LLM 호출 없이 사용자에게"]),
        "rt": (760, 300, 220, 90, "router", "⑤ 라우터 (5.2절)", ["ModernBERT 난이도 분류", "easy / medium / hard"]),
        "llm": (500, 300, 200, 90, "llm", "⑥ Claude 호출", ["Haiku · Sonnet · Opus"]),
        "save": (240, 300, 200, 90, "spring", "⑦ 답 저장", ["exact 키 + 임베딩", "다음 질문의 캐시 후보"]),
    }
    E = [("ui", "ex", "r", "l", "질문", "k"), ("ex", "emb", "r", "l", "miss", "k"), ("emb", "vec", "r", "l", "", "k"),
         ("vec", "nli", "r", "l", "후보 있음", "k"), ("nli", "hit", "b", "t", "통과", "g"),
         ("nli", "rt", "b:0.15", "t:0.85", "불통과@1010,259", "r"), ("vec", "rt", "b", "t", "후보 없음", "k"),
         ("rt", "llm", "l", "r", "", "k"), ("llm", "save", "l", "r", "", "k"), ("save", "ui", "l", "b", "답변@135,345", "k"),
        ]
    build("a1-request-cache", 1300, 470, "요청 처리와 캐시 판정: 같은 질문은 바로, 비슷한 질문은 판정을 거쳐 재사용한다",
          "Spring 게이트웨이와 FastAPI 추론 서버가 나눠 맡는 경로 · ①~④가 캐시, ⑤~⑦이 캐시 miss일 때의 경로",
          n, E, groups=[(220, 86, 1070, 140, "캐시 경로"), (220, 278, 780, 134, "캐시 miss 경로")])


# ---------------------------------------------------------------- A2 escalation
def a2():
    n = {
        "q": (30, 120, 170, 76, "ui", "질문", ["캐시 miss"]),
        "rt": (260, 112, 220, 92, "router", "라우터 판단", ["easy → Haiku", "medium → Sonnet", "hard → Opus"]),
        "t1": (560, 100, 170, 60, "llm", "Haiku", ["가격 1"]),
        "t2": (560, 190, 170, 60, "llm", "Sonnet", ["가격 3"]),
        "t3": (560, 280, 170, 60, "llm", "Opus", ["가격 5"]),
        "ans": (800, 172, 180, 96, "ui", "사용자에게 답변", ["만족하면 종료", "불만족이면 한 단계 위로"]),
        "fb": (1050, 160, 240, 120, "ml", "피드백 이벤트", ["처음 단계 · 최종 단계", "만족 여부 · 상향 횟수", "최종 > 처음이고 만족한", "경우만 학습에 사용"]),
    }
    E = [("q", "rt", "r", "l", "", "k"), ("rt", "t1", "r:0.2", "l", "", "k"), ("rt", "t2", "r:0.5", "l", "", "k"),
         ("rt", "t3", "r:0.8", "l", "", "k"), ("t1", "ans", "r", "l:0.2", "", "k"), ("t2", "ans", "r", "l:0.5", "", "k"),
         ("t3", "ans", "r", "l:0.8", "", "k"), ("ans", "fb", "r", "l", "상향 뒤 만족", "o"),
         ("t1", "t2", "b:0.8", "t:0.8", "상향@745,175", "rd"), ("t2", "t3", "b:0.8", "t:0.8", "상향@745,265", "rd")]
    build("a2-escalation", 1320, 430, "에스컬레이션: 틀리면 사용자가 한 단계 위 모델로 다시 묻고, 그 기록이 재학습 피드백이 된다",
          "단계 가격 비율 1:3:5 · 피드백은 \"더 높은 단계가 필요했다\"는 방향만 담는다",
          n, E)


# ---------------------------------------------------------------- A3 retraining pipeline
def a3():
    n = {
        "sp": (30, 110, 170, 80, "spring", "Spring", ["피드백 묶음 전송", "POST /escalate"]),
        "fa": (250, 102, 210, 96, "api", "FastAPI 접수", ["형식 · 중복 · 만족 검증", "정규화 · SHA-256", "같은 묶음 재전송은 기존 실행"]),
        "gcs1": (510, 110, 170, 80, "store", "GCS", ["피드백 파일 · manifest"]),
        "prep": (730, 102, 210, 96, "ml", "준비 (CPU)", ["이번 cycle 피드백", "+ 과거 피드백 일부", "+ 초기 데이터 일부"]),
        "train": (990, 102, 220, 96, "ml", "학습 (Vertex AI T4)", ["기준점(.0) 가중치에서 시작", "ModernBERT 3 epoch"]),
        "eval": (990, 270, 220, 96, "ml", "평가 · 품질 검사", ["고정 하한 9개", "직전 모델 대비 회귀"]),
        "pkg": (730, 270, 210, 96, "ml", "패키지 · 승격", ["모델 · 라벨 순서 · checksum", "승인 → CURRENT 갱신"]),
        "cur": (510, 278, 170, 80, "store", "GCS CURRENT", ["운영 모델 포인터"]),
        "load": (250, 270, 210, 96, "api", "FastAPI 적재", ["60초마다 확인", "검증 · 완전히 로드한 뒤 교체", "실패하면 기존 유지"]),
        "keep": (990, 420, 220, 56, "note", "거절 → 운영 모델 유지", []),
        "rb": (250, 430, 470, 70, "note", "롤백", ["보호된 이전 승인 모델로 포인터를 되돌림 (현재 · 직전 · 기준점 모델은 정리 대상에서 보호)"]),
    }
    E = [("sp", "fa", "r", "l", "", "k"), ("fa", "gcs1", "r", "l", "", "k"), ("gcs1", "prep", "r", "l", "", "o"),
         ("prep", "train", "r", "l", "", "o"), ("train", "eval", "b", "t", "", "o"), ("eval", "pkg", "l", "r", "통과", "g"),
         ("pkg", "cur", "l", "r", "", "o"), ("cur", "load", "l", "r", "", "k"), ("eval", "keep", "b", "t", "거절", "rd"),
         ("rb", "cur", "t:0.6", "b", "", "kd")]
    build("a3-retraining", 1240, 530, "재학습 파이프라인: 피드백이 쌓이면 학습 → 검사 → 승격 → 서빙 반영까지 자동으로 돈다",
          "Vertex AI 관리형 실행 · 실험 기간 중 품질 검사 없이 올리는 직접 승격 모드도 추가했다 (3.4절)",
          n, E)


# ---------------------------------------------------------------- A4 version policy
def a4():
    n = {
        "v10": (40, 150, 120, 64, "router", "1.0", ["초기 학습"]),
        "v11": (230, 100, 130, 64, "ml", "1.1", ["B1"]),
        "v12": (230, 200, 130, 64, "ml", "1.2", ["B1 + B2"]),
        "v20": (430, 150, 130, 64, "router", "2.0", ["B1 + B2 + B3"]),
        "v21": (630, 100, 130, 64, "ml", "2.1", ["B4"]),
        "v22": (630, 200, 130, 64, "ml", "2.2", ["B4 + B5"]),
        "v30": (830, 150, 130, 64, "router", "3.0", ["B4 + B5 + B6"]),
        "v60": (1060, 150, 170, 64, "note", "… 6.0", ["15번째 성공: 처음 상태에서"]),
    }
    E = [("v10", "v11", "r", "l", "", "o"), ("v10", "v12", "r", "l", "", "o"), ("v10", "v20", "r", "l", "", "o"),
         ("v20", "v21", "r", "l", "", "o"), ("v20", "v22", "r", "l", "", "o"), ("v20", "v30", "r", "l", "", "o"),
         ("v30", "v60", "r", "l", "", "kd")]
    notes = [(40, 300, 1190, "<b>읽는 법</b> &nbsp;화살표는 학습을 시작한 가중치다. 1.1과 1.2는 모두 1.0에서 시작하며, 1.2는 1.1을 이어 학습하지 않는다. "
              "B는 재학습에 성공한 피드백 묶음이고, 같은 cycle 안에서는 묶음을 누적한다. 세 번째 성공이 다음 기준점(.0)이 된다. "
              "실패 · 거절 · 취소는 버전을 올리지 않는다."),
             (40, 360, 1190, "<b>이 규칙이 막는 것</b> &nbsp;① 재학습을 이어 붙일 때 생기는 오차 누적 (항상 검증된 기준점에서 출발) "
              "② 한두 건의 적은 피드백에 휘둘리는 갱신 (cycle 안에서 누적) ③ 오래 쌓인 변화 (다섯 번째 기준점마다 처음 상태의 ModernBERT에서 다시 학습)")]
    build("a4-version-policy", 1260, 420, "버전 정책: 재학습은 항상 마지막 기준점(.0)에서 출발하고, 세 번 성공하면 새 기준점이 된다",
          "운영 기본값 · 과거 피드백은 라벨별 최대 32건, 초기 데이터는 라벨별 32행을 함께 섞는다",
          n, E, notes=notes)


# ---------------------------------------------------------------- A0 overview
def a0():
    n = {
        "ui": (30, 150, 160, 80, "ui", "채팅 UI", ["React"]),
        "sp": (260, 120, 230, 140, "spring", "Spring 게이트웨이", ["대화 · 인증", "exact 캐시 · pgvector", "Claude 호출", "피드백 수집"]),
        "fa": (660, 120, 250, 140, "api", "FastAPI 추론 서버", ["BGE 임베딩", "NLI 재사용 판정", "ModernBERT 라우터", "피드백 접수 · 모델 적재"]),
        "llm": (260, 330, 230, 80, "llm", "Claude", ["Haiku · Sonnet · Opus"]),
        "ml": (1030, 120, 270, 140, "ml", "재학습 (Vertex AI · GCS)", ["준비 → T4 학습 → 검사", "패키지 → CURRENT 갱신", "버전 정책 · 롤백"]),
    }
    E = [("ui", "sp", "r", "l", "", "k"), ("sp", "fa", "r:0.3", "l:0.3", "판정 · 분류 요청", "k"),
         ("fa", "sp", "l:0.7", "r:0.7", "결과", "k"), ("sp", "llm", "b", "t", "단계별 호출", "p"),
         ("fa", "ml", "r:0.3", "l:0.3", "피드백 묶음", "o"), ("ml", "fa", "l:0.7", "r:0.7", "새 모델", "o")]
    build("a0-overview", 1330, 450, "전체 구조: 캐시와 라우터가 요청을 처리하고, 피드백이 재학습을 거쳐 라우터로 돌아온다",
          "세부는 그림 3-2(캐시 경로), 3-3(에스컬레이션), 3-4(재학습), 3-5(버전 정책)",
          n, E)


if __name__ == "__main__":
    for f in [a0, a1, a2, a3, a4]:
        f()
    print("ok")
