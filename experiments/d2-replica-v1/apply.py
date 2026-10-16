"""edits_*.jsonl 을 원문에 적용해 data/dataset.jsonl 을 만든다. 원래 문구는 원문에 정확히 한 번 나와야 한다."""
import glob, json, sys, difflib
orig = {json.loads(l)["gid"]: json.loads(l) for l in open("data/originals.jsonl", encoding="utf-8")}
edits = {}
for f in sorted(glob.glob("data/edits_*.jsonl")):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            e = json.loads(l); edits[e["gid"]] = e
errors, out = [], []
def apply(text, ops, gid, kind):
    for old, new in ops:
        n = text.count(old)
        if n != 1:
            errors.append(f"{gid} {kind}: '{old[:50]}' 등장 {n}회"); return None
        text = text.replace(old, new)
    return text
for gid, o in orig.items():
    if gid not in edits: continue
    e = edits[gid]
    para = apply(o["original"], e["para"], gid, "para"); trap = apply(o["original"], e["trap"], gid, "trap")
    if para is None or trap is None: continue
    if para == o["original"] or trap == o["original"] or para == trap:
        errors.append(f"{gid}: 변형이 원문과 같음"); continue
    ratio = lambda a, b: 1 - difflib.SequenceMatcher(None, a.split(), b.split()).ratio()
    out.append({**o, "para": para, "trap": trap, "trap_type": e["trap_type"], "why": e["why"],
                "para_changed": round(ratio(o["original"], para), 3), "trap_changed": round(ratio(o["original"], trap), 3)})
with open("data/dataset.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"적용 {len(out)} / 원질문 {len(orig)} | 오류 {len(errors)}")
for e in errors: print("  ", e)
if out:
    import statistics as st
    print("바뀐 단어 비중 중앙값: 같은 뜻", st.median(r["para_changed"] for r in out), "/ 함정", st.median(r["trap_changed"] for r in out))
