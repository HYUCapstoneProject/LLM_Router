CSS = """<!doctype html><html><head><meta charset="utf-8"><style>
:root{--ink:#24262b;--mute:#62666f;--line:#c9ccd2;--acc:#4a46b8;--accbg:#f3f2fc;--red:#c2412d;--grn:#127a73}
html,body{margin:0;background:#fff}
body{width:1292px;height:1292px;font-family:"Apple SD Gothic Neo","Noto Sans KR",sans-serif;color:var(--ink);position:relative;word-break:keep-all}
.grid{position:absolute;left:32px;top:32px;display:grid;grid-template-columns:600px 600px;grid-template-rows:600px 600px;gap:28px}
.card{border:2px solid var(--line);border-radius:28px;box-sizing:border-box;position:relative}
.card h2{position:absolute;left:30px;top:26px;margin:0;font-size:24px;line-height:32px}
.ex{position:absolute;left:30px;top:64px;font-size:14px;color:var(--mute)}
.tag{position:absolute;right:28px;top:28px;font-size:13px;font-weight:700;color:var(--acc);border:1.5px solid var(--acc);border-radius:13px;padding:3px 11px}
.vis{position:absolute;left:30px;right:30px;top:104px;height:300px;background:var(--accbg);border:1.6px solid var(--acc);border-radius:16px;box-sizing:border-box;overflow:hidden}
.bottom{position:absolute;left:30px;right:30px;top:428px;height:144px;display:flex;align-items:center;gap:16px}
.rule{flex:1;height:120px;border:1.6px dashed #9aa0aa;border-radius:14px;box-sizing:border-box;padding:14px 16px;font-size:15.5px;line-height:23px;display:flex;align-items:center}
.arrow{width:30px;height:0;border-top:2.4px solid #6b7280;position:relative}
.arrow:after{content:"";position:absolute;right:-2px;top:-8px;border-left:12px solid #6b7280;border-top:7px solid transparent;border-bottom:7px solid transparent}
.pick{width:140px;height:74px;border-radius:14px;background:var(--acc);color:#fff;display:flex;align-items:center;justify-content:center;text-align:center;font-size:18px;font-weight:700;line-height:23px}
svg text{font-family:"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
</style></head><body data-w="1292" data-h="1292"><div class="grid">"""
def card(title, ex, tag, svg, rule, pick):
    t = f'<span class="tag">{tag}</span>' if tag else ''
    return f'''<div class="card"><h2>{title}</h2><div class="ex">{ex}</div>{t}
<div class="vis"><svg width="536" height="296" viewBox="0 0 536 296">{svg}</svg></div>
<div class="bottom"><div class="rule">{rule}</div><div class="arrow"></div><div class="pick">{pick}</div></div></div>'''
A='#4a46b8'; G='#9aa0aa'; R='#c2412d'; K='#24262b'
def dot(x,y,c,lab,dy=26,bold=False):
    w='700' if bold else '400'
    return f'<circle cx="{x}" cy="{y}" r="9" fill="{c}"/><text x="{x}" y="{y+dy}" font-size="14" text-anchor="middle" font-weight="{w}" fill="{c}">{lab}</text>'
def box(x,y,w,h,lab,fill='#fff',stroke=G,fs=15,col=K,bold=False):
    b='700' if bold else '400'
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/><text x="{x+w/2}" y="{y+h/2+5}" font-size="{fs}" text-anchor="middle" font-weight="{b}" fill="{col}">{lab}</text>'
def arr(x1,y1,x2,y2,c='#6b7280',dash=''):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="2.2"{d} marker-end="url(#m)"/>'
MK='<defs><marker id="m" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#6b7280"/></marker></defs>'

# ---------- semantic cache ----------
c1 = MK+f'''<circle cx="268" cy="150" r="96" fill="none" stroke="{A}" stroke-width="2" stroke-dasharray="7 5"/>
<text x="268" y="40" font-size="14" text-anchor="middle" fill="{A}" font-weight="700">거리 기준(예: 코사인 0.80)</text>
{dot(268,150,K,'저장된 질문',bold=True)}<circle cx="222" cy="112" r="9" fill="{A}"/><text x="160" y="96" font-size="14" text-anchor="end" fill="{A}">같은 뜻: 재사용 ✓</text><line x1="164" y1="98" x2="212" y2="108" stroke="{A}" stroke-width="1.4"/>
<circle cx="322" cy="196" r="9" fill="{R}"/><text x="380" y="252" font-size="14" fill="{R}">함정도 안쪽: 재사용 ✗</text><line x1="330" y1="204" x2="384" y2="238" stroke="{R}" stroke-width="1.4"/>'''
c2 = MK+f'''{box(20,60,150,48,'새 질문')}{box(20,170,150,48,'가장 가까운 저장 질문')}
{arr(172,84,236,128)}{arr(172,194,236,152)}
{box(240,112,130,56,'판정 모델',fill='#fff',stroke=A,bold=True,col=A)}
{arr(372,140,410,140)}
<rect x="414" y="90" width="26" height="100" rx="5" fill="#e3e5e9"/><rect x="414" y="104" width="26" height="86" rx="5" fill="{A}"/>
<line x1="406" y1="130" x2="448" y2="130" stroke="{K}" stroke-width="2.2" stroke-dasharray="5 4"/>
<text x="456" y="134" font-size="13" fill="{K}">기준</text><text x="427" y="212" font-size="14" text-anchor="middle" font-weight="700">0.92</text>
<text x="268" y="262" font-size="13.5" text-anchor="middle" fill="#62666f">두 질문을 함께 읽고 점수를 낸다</text>'''
c3 = MK+f'''<circle cx="140" cy="140" r="78" fill="none" stroke="{A}" stroke-width="2" stroke-dasharray="7 5"/>
<circle cx="330" cy="110" r="40" fill="none" stroke="{A}" stroke-width="2" stroke-dasharray="7 5"/>
<circle cx="420" cy="210" r="60" fill="none" stroke="{A}" stroke-width="2" stroke-dasharray="7 5"/>
{dot(140,140,K,'항목 A: 넓은 기준')}{dot(330,110,K,'항목 B: 좁은 기준',dy=-50)}{dot(420,210,K,'항목 C')}
<text x="268" y="282" font-size="13.5" text-anchor="middle" fill="#62666f">틀린 재사용이 생긴 항목은 기준을 좁힌다</text>'''
c4 = MK+f'''<text x="132" y="34" font-size="14" text-anchor="middle" font-weight="700" fill="{K}">일반 임베딩</text>
<text x="404" y="34" font-size="14" text-anchor="middle" font-weight="700" fill="{A}">캐시 전용 임베딩</text>
<line x1="268" y1="20" x2="268" y2="280" stroke="#c9ccd2" stroke-width="1.5"/>
{dot(132,140,K,'원질문',dy=-16)}{dot(100,190,A,'같은 뜻')}{dot(170,196,R,'함정',dy=42)}
{dot(404,120,K,'원질문',dy=-16)}{dot(380,166,A,'같은 뜻')}{dot(490,250,R,'함정',dy=-18)}
<text x="132" y="270" font-size="13" text-anchor="middle" fill="#62666f">둘 다 가깝다</text>
<text x="380" y="270" font-size="13" text-anchor="middle" fill="#62666f">함정은 멀어진다</text>'''
open('w3-cache-types.html','w').write(CSS+
 card('① 임베딩 거리 기준','GPTCache 기본 · 코사인 임계값','',c1,'저장된 질문과의 거리가<br>기준 안이면 답을 재사용','거리로<br>재사용')+
 card('② 판정 모델 추가','GPTCache + 리랭커 · 우리 8월 구성','우리: 리랭커 → NLI',c2,'거리로 후보를 찾고,<br>판정 점수가 기준 이상이면 재사용','판정으로<br>재사용')+
 card('③ 항목별 기준 학습','vCache','',c3,'저장된 항목마다 기준을<br>운영 중에 따로 학습','항목별<br>기준')+
 card('④ 캐시 전용 임베딩','Redis LangCache','',c4,'임베딩을 미세조정해<br>같은 뜻은 가깝게, 다른 뜻은 멀게','거리로<br>재사용')+'</div></body></html>')

# ---------- feedback routing ----------
def armbars(sel, vals, labs, obs):
    s=''
    for i,(v,l,o) in enumerate(zip(vals,labs,obs)):
        x=90+i*130; h=int(v*150)
        c = A if i==sel else G
        s+=f'<rect x="{x}" y="{220-h}" width="60" height="{h}" rx="6" fill="{c}"/><text x="{x+30}" y="244" font-size="14" text-anchor="middle">{l}</text><text x="{x+30}" y="{210-h}" font-size="15" text-anchor="middle" font-weight="700" fill="{A if i==sel else "#62666f"}">{o}</text>'
    return s
f1 = MK+armbars(1,[.55,.75,.6],['Haiku','Sonnet','Opus'],['','✓ +1',''])+f'''
<text x="268" y="34" font-size="14" text-anchor="middle" font-weight="700" fill="{A}">모델별 점수</text>
<text x="268" y="276" font-size="13.5" text-anchor="middle" fill="#62666f">고른 모델의 결과만 보고 그 점수를 고친다 · 가끔 다른 모델도 시도</text>'''
f2 = MK+f'''<text x="268" y="34" font-size="14" text-anchor="middle" font-weight="700" fill="{A}">한 질문에서 보이는 결과</text>
{box(40,60,140,60,'Haiku ✓ (고름)',fill='#fff',stroke=A,bold=True,col=A)}{box(198,60,140,60,'Sonnet ?')}{box(356,60,140,60,'Opus ?')}
{arr(110,124,110,170)}{box(30,174,476,56,'고를 확률이 낮았던 결과일수록 크게 반영(가중치 보정)',fill='#fff',stroke=A,fs=14.5)}
<text x="268" y="266" font-size="13.5" text-anchor="middle" fill="#62666f">"자주 고른 모델만 보인다"는 치우침을 보정해 학습</text>'''
f3 = MK+f'''<text x="268" y="34" font-size="14" text-anchor="middle" font-weight="700" fill="{A}">모델 점수표 (라우터 본체는 그대로)</text>
{box(110,64,150,44,'Haiku')}{box(276,64,150,44,'1,012 → 1,020',col=A,bold=True)}
{box(110,120,150,44,'Sonnet')}{box(276,120,150,44,'1,050 → 1,046')}
{box(110,176,150,44,'Opus')}{box(276,176,150,44,'1,080')}
<text x="268" y="262" font-size="13.5" text-anchor="middle" fill="#62666f">새 결과가 들어오면 표만 고친다 · 새 모델도 표에 추가</text>'''
f4 = MK+f'''{box(20,40,120,52,'사용자')}{box(206,40,150,52,'Opus 답',stroke=A,col=A,bold=True)}
<line x1="204" y1="66" x2="146" y2="66" stroke="#6b7280" stroke-width="2.2" marker-end="url(#m)"/>
{box(206,160,150,52,'Haiku (뒤에서 실행)')}
{arr(281,94,281,156,dash='5 4')}
{box(390,96,130,64,'두 답 비교',stroke=A,col=A,bold=True)}
{arr(358,66,388,110)}{arr(358,186,388,150)}
<text x="455" y="196" font-size="14" text-anchor="middle" font-weight="700" fill="{'#127a73'}">"Haiku도 충분"</text>
<text x="268" y="262" font-size="13.5" text-anchor="middle" fill="#62666f">사용자는 강한 모델의 답을 받고, 비교 결과는 학습 데이터가 된다</text>'''
open('w4-feedback-types.html','w').write(CSS+
 card('① 탐색하며 배우기 (bandit)','PILOT · MixLLM','',f1,'고른 모델의 결과로 점수를 갱신하고<br>가끔 다른 모델도 시도','점수<br>갱신')+
 card('② 반사실 보정','Causal LLM Routing','',f2,'보이지 않은 모델의 결과를<br>가중치로 보정해 학습','치우침<br>보정')+
 card('③ 학습 없이 갱신','Eagle · UniRoute','',f3,'라우터는 다시 학습하지 않고<br>모델 점수표만 고침','즉시<br>반영')+
 card('④ 그림자 실행','업계 운영 방식','',f4,'강한 모델 답을 내보내며<br>싼 모델을 뒤에서 돌려 비교','양방향<br>정답')+'</div></body></html>')
