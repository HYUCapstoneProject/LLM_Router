# 성능 평가 색인

> **2026-10-01 데이터 오류 경고:** RouterBench·xRouteBench의 기록 점수에서 체계적인 채점 오류를 발견했다(RouterBench GSM8K 등급 점수, MMLU 정답 키 오류, xRouteBench 형식 오답 처리, SQuAD F1 기준). 아래 2026-09 실험은 모두 오류가 포함된 v1 데이터 기준이다. 현재 기준은 v1을 유지하며, v2는 교체용으로 보관한다(재실험 여부 보류). → [점검 보고서와 검증 데이터셋 v2](benchmark-outcome-audit-20261001/REPORT.md)

2026-10-05 추가: [캐시 실험 C0~C3 run001](cache-experiment-v1-20261005/REPORT.md). exact 캐시는 잘못된 재사용 0이다. 의미 캐시(운영 C3 포함)는 D2 test에서 hit 중 45~50%가 다른 질문의 답이었다(LLM 판정 기준).

2026-09-25 추가: [R1/R3 natural·balanced 최종 평가](routing-generalization-v1-20260925/report.md), [R0~R3 통합 판단](routing-method-review-20260925/REVIEW.md), [Git·원본 보관 범위](ARTIFACT_STORAGE.md).

현재 최종 설계에 따른 실행 기록은 [R0 초기 학습·복구](modernbert-initial-v1-20260924/REPORT.md), [R0/R1/R2 통합 A 비교](routing-objectives-continual-v1-A-20260924/REPORT.md), [비교 결과 해석](routing-objectives-continual-v1-A-20260924/ANALYSIS.md)에 있다. 세 방법을 같은 2,007 step·seed 42로 비교했으며 D_dev 비용–품질 nAUC는 R1 0.863179, R2 0.833874, R0 0.166115로 R1이 선정됐다. GCS·개인 Drive 보존을 검증했다. 공개 후보의 단일 seed 결과이며, [독립 test 7,202문항 평가](routing-objectives-continual-v1-A-20260924/sealed-tests/REPORT.md)도 완료했으며, R1/R2의 일부 source 비용 상한 초과를 기록했다. 실제 Claude 서비스 검증·운영 활성화는 별도다. 이후 실험은 [공통 보고서 형식](../templates/RUN_REPORT.md)으로 데이터·학습 조건, 비교·선정, 결과·해석, 파일 보존·게시 상태를 기록한다.

아래는 2026년 8월에 수행한 기존 서비스·라우터 측정의 색인이다. 각 문서에 방법·원본 데이터·한계가 있다.
숫자를 인용하기 전에 **"인용 시 주의"** 열을 먼저 볼 것.

기준일: 2026-08-30

2026-09-23 통합 시 기존 REPORT·CSV·JSON·그림과 재현 도구를 보존했다. 아래 수치는 [ML 최종 실험 명세](../../docs/ml/README.md)의 RB+xRB 공통 split·후보·학습 목표·캐시·feedback 실험 결과가 아니다. 현재 설계와 [구현 차이](../../docs/ml/README.md#implementation-gaps), [변경 이력](../../docs/deprecated-settings.md)을 우선한다. 당시 배포 상태도 현재 live 상태로 간주하지 않는다.

기존 파레토의 random은 **11개 모델 균등 선택**이며 최신 호출 비율 일치 random과 다르다. `+0.0119`의 배정은 Yi/Yi/gpt-4로 실제 두 모델을 사용한다. 서로 다른 세 모델 배정은 `+0.0079`다. 이 관측들을 최신 세 후보 공정 비교의 완료 증거로 옮기지 않는다.

## 1. 한눈에

| # | 측정 | 문서 | 데이터 | Claude 비용 |
|---|---|---|---|---|
| 1 | 임베딩 계약과 캐시 임계값 | [cache-threshold](cache-threshold/REPORT.md) | 자체 질문 92문항 | 0 |
| 2 | 리랭커 모델 선정 (4종) | [reranker-20260830](reranker-20260830/REPORT.md) | 58쌍 (위에서 파생) | 0 |
| 3 | 리랭커 실적용 검증 | [rerank-endtoend](rerank-endtoend/REPORT.md) | 자체 질문 92문항 | 0 |
| 4 | 라우터 분류 성능 | [router-classification](router-classification/REPORT.md) | RouterBench 3,218 + 자체 35 | 0 |
| 5 | 기존 기법 대비 파레토 | [pareto](pareto/REPORT.md) | RouterBench 36,497 | 0 |
| 6 | exact match 캐시 | [exact-match](exact-match/REPORT.md) | 운영 스택 실측 3건 | 미미 |
| 7 | 티어 재현율 (배정 없는 평가 시도) | [tier-agreement](tier-agreement/REPORT.md) | 위 5번 예측 재사용 | 0 |
| — | GPTCache 구현 비교 | [../GPTCACHE_COMPARISON.md](../GPTCACHE_COMPARISON.md) | 문헌·코드 조사 | 0 |

**일곱 측정 중 여섯이 Claude 를 한 번도 부르지 않았다.** RouterBench 에 11개 모델의
정오답과 비용이 들어 있고, 임베딩·리랭커·분류는 자체 serving에서 추론했기 때문이다.
표의 비용은 신규 Claude 호출비이며, 서버·추론·저장 비용까지 0이라는 뜻은 아니다.

## 2. 핵심 수치와 인용 시 주의

| 수치 | 값 | ⚠️ 인용 시 주의 |
|---|---|---|
| 파레토 프런티어 대비 마진 | **+0.0119** (95% CI [+0.0040, +0.0203]) | 우리 *분류기*를 RouterBench 모델에 이식한 값. **우리 서비스를 잰 게 아니다** |
| 리랭커 적용 시 오탐 | **19건 → 1건** | 오탐률 36.5%→1.9% 는 **약점을 노려 만든 질문셋** 기준. 절대값 인용 금지, 비교값만 사용 |
| 리랭커 적용 시 recall | 100% → **90.5%** | 잃은 2건은 Claude 를 한 번 더 부를 뿐 사용자는 정상 답변을 받는다 |
| 라우터 라벨 일치율 | **47.5%** (RouterBench) / **31.4%** (개방형) | 두 값이 다른 이유가 핵심이다. 하나만 인용하면 오해 |
| exact match 지연 | 15,063ms → **5ms** | 아낀 것은 embed 왕복(120ms)과 LLM 호출. "15초를 5ms로 줄였다"는 표현은 부정확 |
| 캐시 임계값 0.80 오탐 | 37 / 4,163쌍 | 분모가 전체 조합이라 실사용 오탐률이 아니다 |
| 티어 재현율 | **29.8% ~ 40.8%** | 묶는 방식·판정 규칙에 따라 무작위(33.3%) 위아래로 갈린다. **단일 값 인용 금지** |

## 3. 측정이 서로 어떻게 이어지는가

```
[1] 임베딩 계약·임계값
     paraphrase 최소 0.8621 < hard_negative 최대 0.9371
     → 어떤 코사인 임계값도 두 클래스를 못 가른다
                    ↓
[2] 리랭커 모델 선정
     4종 오프라인 스크리닝 → quora-distilroberta + BGE AND 조합
     사전 기준(오탐 0 + recall 90%)은 통과 실패
                    ↓
[3] 리랭커 실적용 검증          ← enforce 를 켠 근거
     오탐 19→1, 정탐 21→19
                    ↓
     운영 mode=enforce (fail-closed)

[4] 라우터 분류 성능 ─────┐
                          ├→ [5] 파레토: 우리 정책이 기준선을 넘는가
RouterBench 원본 ─────────┘        프런티어 대비 +0.0119
                                   부수 발견: 병목은 라벨 정의

[6] exact match — 위와 독립. 동일 질문에 embed 왕복을 없앤다
```

## 4. 아직 측정하지 않은 것

| 항목 | 왜 필요한가 | 막고 있는 것 |
|---|---|---|
| **실험 A — 우리 시스템 ablation** | 파레토는 우리 *분류기*만 잰 것이다. Claude 모델·캐시를 포함한 실제 절감률이 없다 | **품질 측정 수단이 없다.** 우리 질문에는 정답 라벨이 없어 "얼마나 잃었나"를 못 잰다. RouterBench 문항을 쓰거나 LLM judge 를 붙여야 한다 |
| RouteLLM 대비 | 선행연구 대비 개선폭 | 공개 체크포인트는 학습 도메인이 달라 불공정. 같은 라벨로 재학습 필요 (팀원) |
| 라벨 재정의 효과 | 파레토에서 천장이 +0.027 → +0.167 로 오른다고 나왔다 | 재학습 필요 (팀원) |
| 리랭커 지연의 운영 영향 | 시맨틱 히트가 200ms → 1,178ms 로 늘었다 | 표본이 적다. 실험 A 와 함께 재면 된다 |
| 캐시가 커졌을 때의 오탐 | 현재 검증은 캐시 19건 기준. 커지면 후보가 늘어 오탐 기회도 는다 | 데이터 축적 필요 |

## 5. 측정하면서 드러난 결함

기록해 둔다. 같은 실수를 반복하지 않기 위해서다.

**(1) Java 15 의 `\s` 이스케이프** — 정규식 `"\s+"` 가 리터럴 스페이스로 해석되어
탭·줄바꿈이 정리되지 않았다. 컴파일은 통과했다. → [exact-match](exact-match/REPORT.md)

**(2) sentence-transformers 버전 불일치** — 스크리닝은 6.0.0, 서빙에는 3.3.1 을
박았다. 로딩 실패로 드러났지만, **조용히 성공했다면 활성함수가 달라져 임계값이
무효가 됐을 것이다.** 이후 `scripts/verify_reranker_parity.py` 로 점수 동등성을
직접 검증한다(58쌍, 최대 차이 2.14e-06, 판정 뒤집힘 0건).

**(3) 파레토 사다리 선택 편향** — 조합을 다 보고 test 에서 최선을 고르면 과적합이다.
validation 에서 고르도록 고쳤다. 격차 +0.0283 → +0.0119 로 절반 이하가 됐다.

**(4) 티어별 임계값 가설 반증** — v1 질문셋(hard negative n=2)에서는 맞아 보였으나
v2(n=37)에서 최악 사례가 세 티어에 고르게 퍼져 있었다. 소표본 결론의 예다.

**(5) "라우터가 easy 로 붕괴한다"는 과잉 일반화였다** — 개방형 질문에서만 관측된
현상이고 RouterBench 에서는 분포가 균형적이다. 두 데이터를 구분해 서술해야 한다.

**(6) "과대 라우팅으로 치우친다"는 서술도 조건부였다** — 그 결론은
"가장 싼 충분 모델" 정의에서만 성립한다. 티어 과반을 요구하는 정의로 바꾸면
과대 29.4% / 과소 36.3% 로 **방향이 뒤집힌다**. 사다리에 강건한 것이지
판정 규칙에 강건한 것이 아니었다. → [tier-agreement](tier-agreement/REPORT.md) §4

**(7) 파레토가 고른 사다리는 easy 와 medium 이 같은 모델이다**(Yi/Yi/gpt-4).
격차 최대화로 고른 결과인데, 그러면 3티어가 실질 2티어로 무너져 배포 구성과 어긋난다.
3모델을 유지하는 사다리(Yi/gpt-3.5/gpt-4)에서는 +0.0079 다. 결론은 유지되나 폭이 준다.

## 6. 재현 환경

```bash
python -m venv .venv-experiments
.venv-experiments/Scripts/python.exe -m pip install pandas pyarrow matplotlib
set ROUTER_API_TOKEN=<토큰>          # 팀원 FastAPI 서빙 토큰
```

RouterBench 원본은 GCS 에 있다.

```
gs://llm-router-mlserver/datasets/routerbench/raw/routerbench_0shot.pkl        (95MB)
gs://llm-router-mlserver/datasets/routerbench/.../training.parquet             (7MB, 라벨·split)
```

`.venv-experiments/` 는 gitignore 대상이다. 각 문서의 "재현" 절에 개별 명령이 있다.
