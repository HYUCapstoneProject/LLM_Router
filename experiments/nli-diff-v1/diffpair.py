"""두 입력에서 공통 문장을 빼고 달라진 문장만 남긴다.

1) 줄바꿈으로 나눈 뒤 각 줄을 문장 끝(. ! ?) 뒤 공백에서 다시 나눈다.
2) difflib 로 문장 목록을 맞춰, 같지 않은 구간의 문장만 모은다.
3) 한쪽이 비면(문장 추가·삭제) 바뀐 구간 바로 앞의 공통 문장(없으면 바로 뒤)을 양쪽에 붙인다.
4) 공통 문장이 하나도 없으면 원문 그대로 둔다.

normalize=True 이면 문장을 맞출 때만 대소문자와 공백을 무시한다(v2). 남기는 문장은 원문 그대로다.
숫자·기호·구두점은 건드리지 않는다 — 차이를 숨길 수 있는 정규화는 하지 않는다.
"""
import difflib
import re

_SENT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")


def segments(text):
    out = []
    for line in text.split("\n"):
        for s in _SENT.split(line.strip()):
            s = s.strip()
            if s:
                out.append(s)
    return out


def _key(s):
    return _WS.sub(" ", s).strip().lower()


def reduce_pair(a, b, normalize=False):
    sa, sb = segments(a), segments(b)
    ka, kb = ([_key(x) for x in sa], [_key(x) for x in sb]) if normalize else (sa, sb)
    ops = difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes()
    if not any(tag == "equal" for tag, *_ in ops):
        return a, b
    da, db = [], []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        da.extend(sa[i1:i2])
        db.extend(sb[j1:j2])
    if not da or not db:
        ctx = []
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                continue
            ctx.append(sa[i1 - 1] if i1 > 0 else (sa[i2] if i2 < len(sa) else None))
        ctx = [c for c in ctx if c]
        da, db = ctx + da, ctx + db
    if not da or not db:
        return a, b
    return " ".join(da), " ".join(db)
