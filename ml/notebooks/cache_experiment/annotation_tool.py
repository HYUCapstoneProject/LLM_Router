"""Self-contained blind annotation tool for d2/annotation-A.csv (or B; they are identical).

Usage: python annotation_tool.py <annotation-A.csv> <out.html>
The page shows one item at a time with a word diff, hides the sheet id, autosaves in
the browser and exports annotation-<slot>-input.csv in the exact sheet format.
"""
import json
import sys

import pandas as pd

src, out = sys.argv[1], sys.argv[2]
sheet = pd.read_csv(src, encoding="utf-8-sig", dtype=str, keep_default_na=False)
COLUMNS = list(sheet.columns)
rows = [{"id": r.item_id, "o": r.original_text, "v": r.variant_text, "a": r.cached_answer, "g": r.original_gold}
        for r in sheet.itertuples()]
data = json.dumps({"columns": COLUMNS, "rows": rows}, ensure_ascii=False).replace("</", "<\\/")

page = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>D2 주석 도구</title>
<style>
:root{--bg:#fbfaf7;--fg:#1f1f1f;--muted:#6b6b6b;--card:#fff;--line:#e4e1da;--ins:#fff1a8;--del:#f6d5cf;--accent:#2f6fb3;--ok:#2f7d4f;--warn:#b3412f}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--fg:#ececec;--muted:#a3a3a3;--card:#242423;--line:#3a3936;--ins:#5c4d10;--del:#5a2a22;--accent:#7fb0e8;--ok:#7fcf9c;--warn:#f08b78}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:19px;margin:0 0 4px}
.muted{color:var(--muted)}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
button,select,input,textarea{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:6px}
button{padding:5px 12px;cursor:pointer} button:disabled{opacity:.4;cursor:default}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
input,select{padding:5px 8px}
.progress{height:6px;background:var(--line);border-radius:3px;overflow:hidden;flex:1;min-width:160px}
.progress>div{height:100%;background:var(--ok)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:760px){.grid{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-wrap:anywhere}
.panel h2{font-size:13px;margin:0 0 6px;color:var(--muted);font-weight:600}
.text{white-space:pre-wrap;font-size:14px}
ins{background:var(--ins);text-decoration:none;border-radius:2px}
del{background:var(--del);border-radius:2px}
.answer{border-left:4px solid var(--accent)}
.q{margin:14px 0 6px;font-weight:600}
.choices{display:flex;gap:8px;flex-wrap:wrap}
.choices button{min-width:110px}
.choices button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
textarea{width:100%;min-height:54px;padding:8px}
.kbd{font-size:12px;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--muted)}
.setup{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;max-width:620px}
.hidden{display:none}
.warn{color:var(--warn)}
details{margin:8px 0}
</style></head><body><main>
<h1>D2 주석 도구</h1>
<div class="muted">변형 질문 600개를 원질문과 비교해 판정합니다. 작성 의도(같은 뜻인지 함정인지)는 숨겨져 있고 순서는 섞여 있습니다. 다른 주석자와 상의하지 마세요.</div>

<details><summary><b>판정 기준</b></summary>
<ul>
<li><b>같은 질문인가?</b> 변형 질문이 원질문과 <b>같은 것을 묻는지</b>. 표현만 다르면 Y, 묻는 내용·조건·수량이 달라지면 N.</li>
<li><b>저장된 답을 그대로 써도 맞는가?</b> 원질문에 저장된 답(파란 상자)을 <b>변형 질문의 답으로 내보내도 맞는지</b>. 같은 뜻이어도 저장된 답 자체가 틀렸으면 N. 판단할 수 없으면 U.</li>
<li>근거는 한 줄이면 충분합니다. 예: "숫자 6→8로 바뀌어 답이 달라짐", "표현만 바뀜".</li>
</ul>
<div class="muted">단축키: <span class="kbd">1</span>/<span class="kbd">2</span> 같은 질문 Y/N · <span class="kbd">Q</span>/<span class="kbd">W</span>/<span class="kbd">E</span> 저장 답 Y/N/U · <span class="kbd">→</span>/<span class="kbd">Enter</span> 다음 · <span class="kbd">←</span> 이전 (근거 입력 중에는 단축키가 꺼집니다)</div>
</details>

<section id="setup" class="setup">
<p><b>시작하기</b></p>
<div class="bar"><label>주석자 슬롯 <select id="slot"><option value="A">A</option><option value="B">B</option></select></label>
<label>이름 <input id="name" placeholder="본인 이름"></label>
<button class="primary" id="start">시작</button></div>
<div class="muted">진행 상황은 이 브라우저에 자동 저장됩니다. 다른 컴퓨터로 옮기거나 이어서 하려면 "CSV 내보내기"로 받은 파일을 "CSV 불러오기"로 다시 읽으면 됩니다.</div>
<div class="bar"><label>이전 작업 불러오기 <input type="file" id="importFile" accept=".csv"></label></div>
</section>

<section id="work" class="hidden">
<div class="bar"><b id="pos"></b><div class="progress"><div id="prog"></div></div><span id="done" class="muted"></span>
<label><input type="checkbox" id="onlyTodo"> 미완료만</label>
<input id="jump" type="number" min="1" style="width:84px" placeholder="번호"><button id="go">이동</button></div>
<div class="grid">
<div class="panel"><h2>원질문</h2><div id="orig" class="text"></div></div>
<div class="panel"><h2>변형 질문 <span class="muted">(원질문과 다른 부분 노란색)</span></h2><div id="vari" class="text"></div></div>
</div>
<div class="panel answer" style="margin-top:12px"><h2>원질문에 저장된 답 <span id="gold" class="muted"></span></h2><div id="ans" class="text"></div></div>
<div class="q">1. 변형 질문이 원질문과 같은 것을 묻나요?</div>
<div class="choices" data-field="semantic_equivalent"><button data-v="Y">Y 같다 <span class="kbd">1</span></button><button data-v="N">N 다르다 <span class="kbd">2</span></button></div>
<div class="q">2. 저장된 답을 변형 질문의 답으로 내보내도 맞나요?</div>
<div class="choices" data-field="cached_answer_correct_for_target"><button data-v="Y">Y 맞다 <span class="kbd">Q</span></button><button data-v="N">N 틀리다 <span class="kbd">W</span></button><button data-v="U">U 판단 불가 <span class="kbd">E</span></button></div>
<div class="q">근거 (한 줄)</div>
<textarea id="why"></textarea>
<div class="bar"><button id="prev">← 이전</button><button id="next" class="primary">다음 →</button>
<span style="flex:1"></span><button id="export">CSV 내보내기</button><span id="exportMsg" class="muted"></span></div>
</section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const ROWS = DATA.rows, N = ROWS.length;
let slot = 'A', name = '', idx = 0, ans = {};
const $ = id => document.getElementById(id);
const key = () => 'd2-annot-' + slot;
function save(){ try { localStorage.setItem(key(), JSON.stringify({name, idx, ans})); } catch(e) {} }
function load(){ try { const s = JSON.parse(localStorage.getItem(key()) || 'null'); if (s) { ans = s.ans || {}; idx = s.idx || 0; if (!name) name = s.name || ''; } } catch(e) {} }
function esc(s){ return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function diff(a, b){
  const A = a.match(/\S+|\s+/g) || [], B = b.match(/\S+|\s+/g) || [];
  const m = A.length, n = B.length;
  if (m * n > 4e6) return esc(b);
  const L = Array.from({length: m + 1}, () => new Uint16Array(n + 1));
  for (let i = m - 1; i >= 0; i--) for (let j = n - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i+1][j+1] + 1 : Math.max(L[i+1][j], L[i][j+1]);
  let i = 0, j = 0, out = '';
  while (i < m && j < n) {
    if (A[i] === B[j]) { out += esc(B[j]); i++; j++; }
    else if (L[i+1][j] >= L[i][j+1]) { if (A[i].trim()) out += '<del>' + esc(A[i]) + '</del>'; i++; }
    else { out += B[j].trim() ? '<ins>' + esc(B[j]) + '</ins>' : esc(B[j]); j++; }
  }
  while (i < m) { if (A[i].trim()) out += '<del>' + esc(A[i]) + '</del>'; i++; }
  while (j < n) { out += B[j].trim() ? '<ins>' + esc(B[j]) + '</ins>' : esc(B[j]); j++; }
  return out;
}
const complete = r => { const x = ans[r.id] || {}; return !!(x.semantic_equivalent && x.cached_answer_correct_for_target); };
function render(){
  const r = ROWS[idx], x = ans[r.id] || {};
  $('pos').textContent = '#' + (idx + 1) + ' / ' + N;
  $('orig').innerHTML = esc(r.o);
  $('vari').innerHTML = diff(r.o, r.v);
  $('ans').textContent = r.a;
  $('gold').textContent = r.g ? '(원질문 정답: ' + r.g + ')' : '';
  document.querySelectorAll('.choices').forEach(c => c.querySelectorAll('button').forEach(b => b.classList.toggle('on', x[c.dataset.field] === b.dataset.v)));
  $('why').value = x.rationale || '';
  const done = ROWS.filter(complete).length;
  $('done').textContent = '완료 ' + done + ' / ' + N;
  $('prog').style.width = (100 * done / N) + '%';
  $('prev').disabled = idx === 0;
  save();
}
function setField(f, v){ const r = ROWS[idx]; ans[r.id] = Object.assign(ans[r.id] || {}, {[f]: v}); render(); }
function move(d){
  const only = $('onlyTodo').checked; let k = idx;
  do { k += d; } while (only && k >= 0 && k < N && complete(ROWS[k]));
  if (k >= 0 && k < N) { idx = k; render(); window.scrollTo(0, 0); }
}
document.querySelectorAll('.choices').forEach(c => c.addEventListener('click', e => { const b = e.target.closest('button'); if (b) setField(c.dataset.field, b.dataset.v); }));
$('why').addEventListener('input', e => { const r = ROWS[idx]; ans[r.id] = Object.assign(ans[r.id] || {}, {rationale: e.target.value}); save(); });
$('prev').onclick = () => move(-1); $('next').onclick = () => move(1);
$('go').onclick = () => { const k = parseInt($('jump').value, 10); if (k >= 1 && k <= N) { idx = k - 1; render(); } };
document.addEventListener('keydown', e => {
  if ($('work').classList.contains('hidden') || ['TEXTAREA','INPUT'].includes(document.activeElement.tagName)) return;
  const m = {'1':['semantic_equivalent','Y'],'2':['semantic_equivalent','N'],'q':['cached_answer_correct_for_target','Y'],'w':['cached_answer_correct_for_target','N'],'e':['cached_answer_correct_for_target','U']}[e.key.toLowerCase()];
  if (m) { setField(...m); e.preventDefault(); }
  else if (e.key === 'ArrowRight' || e.key === 'Enter') { move(1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { move(-1); e.preventDefault(); }
});
function csvCell(v){ v = String(v ?? ''); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
$('export').onclick = () => {
  const todo = ROWS.length - ROWS.filter(complete).length;
  if (todo && !confirm('아직 ' + todo + '개가 미완료입니다. 중간 저장용으로 내보낼까요? (02 병합에는 600개 모두 필요합니다)')) return;
  const lines = [DATA.columns.join(',')];
  for (const r of ROWS) {
    const x = ans[r.id] || {};
    const rec = {item_id: r.id, original_text: r.o, variant_text: r.v, cached_answer: r.a, original_gold: r.g,
                 semantic_equivalent: x.semantic_equivalent || '', cached_answer_correct_for_target: x.cached_answer_correct_for_target || '',
                 rationale: x.rationale || '', annotator: name};
    lines.push(DATA.columns.map(c => csvCell(rec[c])).join(','));
  }
  const blob = new Blob(['﻿' + lines.join('\r\n') + '\r\n'], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'annotation-' + slot + '-input.csv'; a.click();
  $('exportMsg').textContent = 'annotation-' + slot + '-input.csv 저장됨' + (todo ? ' (미완료 ' + todo + ')' : '');
};
function parseCSV(t){
  const out = []; let row = [], f = '', q = false;
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (q) { if (c === '"') { if (t[i+1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n') { row.push(f.replace(/\r$/, '')); out.push(row); row = []; f = ''; }
    else f += c;
  }
  if (f || row.length) { row.push(f); out.push(row); }
  return out;
}
$('importFile').onchange = async e => {
  const file = e.target.files[0]; if (!file) return;
  const rows = parseCSV((await file.text()).replace(/^﻿/, ''));
  const h = rows.shift(), ix = Object.fromEntries(h.map((c, i) => [c, i]));
  const ids = new Set(ROWS.map(r => r.id)); let n = 0;
  for (const r of rows) {
    const id = r[ix.item_id]; if (!ids.has(id)) continue;
    ans[id] = {semantic_equivalent: (r[ix.semantic_equivalent] || '').trim().toUpperCase(),
               cached_answer_correct_for_target: (r[ix.cached_answer_correct_for_target] || '').trim().toUpperCase(),
               rationale: r[ix.rationale] || ''};
    if (!name && r[ix.annotator]) name = r[ix.annotator];
    n++;
  }
  alert(n + '개 항목을 불러왔습니다. 슬롯과 이름을 확인하고 시작을 누르세요.');
  if (name) $('name').value = name;
};
$('start').onclick = () => {
  const imported = ans; slot = $('slot').value; name = $('name').value.trim();
  if (!name) { alert('이름을 입력하세요.'); return; }
  load(); if (Object.keys(imported).length) ans = Object.assign(ans, imported);
  name = $('name').value.trim();
  $('setup').classList.add('hidden'); $('work').classList.remove('hidden'); render();
};
</script></body></html>'''
page = page.replace("__DATA__", data)
open(out, "w").write(page)
print(out, len(page) // 1024, "KB", len(rows), "rows")
