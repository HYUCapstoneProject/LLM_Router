# 실험 목록과 코드

기준일: 2026-10-16. 이 프로젝트에서 수행했거나 계획한 실험을 코드 하나로 부른다. 대화·문서·PR·보고서에서 실험을 가리킬 때는 이 코드를 쓴다.

## 1. 코드 규칙

| 영역 | 뜻 | 예 |
|---|---|---|
| `DT` | 데이터 구축·감사 | DT-02 채점 점검과 v2 |
| `RT` | 라우터 학습·선정·추론 조건 | RT-05 R0/R1/R2 비교 |
| `RF` | 재학습(피드백) | RF-03 F4 |
| `EV` | 평가·분석 | EV-04 천장 분석 |
| `CA` | 캐시 | CA-05 C0~C3 run001 |
| `SY` | 시스템·운영 검증 | SY-02 직접 승격 E2E |
| `XA` | 대화 중 탐색 분석(정식 실험 아님) | XA-03 세대 간 전이 |

- 번호는 영역마다 시간 순서로 붙이고 다시 쓰지 않는다. 계획 단계에서 미리 코드를 받는다.
- **실험 코드와 비교군 이름은 다르다.** R0~R3(학습 목표), F0~F5(재학습 비교군), C0~C3(캐시 비교군), E1~E4(EV/RT 계획 안의 실험 이름)는 실험 안의 비교군·단계 이름으로 그대로 쓴다.
- 새 결과 폴더는 `experiments/results/<코드 소문자>-<요약>-<YYYYMMDD>/`로 만들고(예: `rt08-v2-retrain-20261020`), 보고서 첫 줄 아래에 `실험 코드:`를 적는다. 기존 폴더 이름은 링크 보존을 위해 바꾸지 않는다.
- 상태 값: **완료**, **완료·PR 검토 중**, **진행**, **계획**, **선택**, **보류**, **통합**, **폐기**, **탐색**. "통합"은 다른 실험 코드 안에서 함께 수행한다는 뜻이다.
- 팀원이 수행한 실험은 이름 뒤에 `(팀원)`을 붙인다. 표시가 없는 실험은 우리(사용자·Claude)가 수행했다.
- 상태가 바뀌면 이 표를 먼저 고친다. 수치는 각 보고서가 원본이며 이 표는 한 줄 요약이다.
- 보관 manifest에 SHA가 고정된 보고서(RT-06, RT-07, EV-02)는 원본 무결성을 위해 본문에 코드를 쓰지 않고 이 표에서만 연결한다.

## 2. 실험 목록

### 2.1. 데이터 (DT)

| 코드 | 실험 | 시기 | 상태 | 핵심 결과 | 기록 |
|---|---|---|---|---|---|
| DT-01 | 통합 데이터·연속학습 분할 v1 (RouterBench + xRouteBench) | 09-24 | 완료 | 42,935행. train 21,406 / D_dev 3,581 / D_gate 3,581 / stream 5,368 / retention 1,797 / 봉인 test 7,202. 난이도 경계 T1 .27/.64 | [ML 명세 2장](../docs/ml/README.md) |
| DT-02 | 벤치마크 채점 점검과 검증 데이터셋 v2 | 10-01 | 완료 | RouterBench 82,519칸·xRouteBench 13,733칸 정정, RouterBench 290문항 제외. GPT-4 GSM8K 기록 68.1% → 실제 95.3%. v2 경계 .36/.73. 학습 기준은 v1 유지 결정 | [보고서](results/benchmark-outcome-audit-20261001/REPORT.md) |
| DT-03 | 외부 평가 데이터 준비 (LLMRouterBench·SPROUT) | 10-01 | 완료 | LLMRouterBench 9,710문항(2025년 모델 13개, 비용 포함), SPROUT 19,458문항(2024년 모델 13개). calibration 30% / test 70%로 준비했고, 10-16 결정으로 EV-10에서 train 50 / dev 15 / test 35로 다시 나눠 일부를 학습에 쓴다 | [문서](../docs/ml/EXTERNAL_EVALUATION_DATA.md), [manifest](results/external-evaluation-data-v1-20261001/manifest.json) |

### 2.2. 라우터 학습·선정 (RT)

| 코드 | 실험 | 시기 | 상태 | 핵심 결과 | 기록 |
|---|---|---|---|---|---|
| RT-01 | 라우터 분류 성능 (팀원, 8월 선행) | 08 | 완료 | 라벨 일치 47.5%(RouterBench 3,218) / 31.4%(자체 35) | [보고서](results/router-classification/REPORT.md) |
| RT-02 | 기존 기법 대비 파레토 (팀원, 8월 선행) | 08 | 완료 | 프런티어 대비 +0.0119 [+0.0040, +0.0203]. 분류기를 RouterBench 모델에 이식한 값이며 서비스 측정이 아님 | [보고서](results/pareto/REPORT.md) |
| RT-03 | 티어 재현율 (팀원, 8월 선행) | 08 | 완료 | 29.8~40.8%. 묶는 방식에 따라 무작위(33.3%) 위아래로 갈림 | [보고서](results/tier-agreement/REPORT.md) |
| RT-04 | R0 초기 학습·복구 | 09-24 | 완료 | ModernBERT-base, A train 21,406, 2,007 step, seed 42. 이후 운영 1.0(SY-01) | [보고서](results/modernbert-initial-v1-20260924/REPORT.md) |
| RT-05 | R0/R1/R2 학습 목표 비교 (통합 A)와 봉인 test | 09-24 | 완료 | D_dev nAUC R1 .863 > R2 .834 > R0 .166. test 품질 R1−R0 +.132(비용 +.40). R1 선정 | [보고서](results/routing-objectives-continual-v1-A-20260924/REPORT.md), [test](results/routing-objectives-continual-v1-A-20260924/sealed-tests/REPORT.md) |
| RT-06 | G1 일반화: R1/R3 × natural/balanced | 09-25 | 완료 | selection은 R1-natural. 처음 보는 과제에서 R3가 품질·비용 우위. xRouteBench에서는 항상 low·medium이 네 방법을 모두 이김 | [보고서](results/routing-generalization-v1-20260925/report.md) |
| RT-07 | 추론 길이 512 → 8,192 | 09-25 | 완료 | 성능 차이 미미(R0 RouterArena 45.48 → 45.39%). 09-26에 운영 8,192 채택 | [보고서](results/context-length-qualification-20260925/REPORT.md) |
| RT-08 | v1 채점 오류의 영향: 기존 정의 R0의 v2 재학습 | 10-16~ | 진행 중(학습 완료, 평가는 EV-10 07b·08) | 기존 정의 R0(경계 .27/.64)를 A 분할 v2로 한 번 재학습해 운영 R0(v1)와 비교. 새 정의와 섞지 않아 채점 효과만 분리 | [최종 계획서 3절](../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md) |

### 2.3. 재학습 (RF)

| 코드 | 실험 | 시기 | 상태 | 핵심 결과 | 기록 |
|---|---|---|---|---|---|
| RF-01 | R1→R0 전이 (A 이어 학습 / B R1 출력층 전이 / C 동결 R1) | 09-27 | 완료 | B가 A보다 낫지 않음(−.006). 둘 다 망각 기준 미달. easy 예측 61% → 14% 상향 편향 | [분석](results/r1-to-r0-transfer-v2-final-only-20260927/ANALYSIS.md), [계획서](../docs/ml/R1_TO_R0_TRANSFER_EXPERIMENT.md) |
| RF-02 | 재학습 효과 F0~F3 (상향 피드백) | 09-29~10-01 | 완료(결과 폴더 미반입) | F1~F3 첫 시도 품질 +.14~.19에 비용 +63~77%, 망각 미달, 미채택. 원본은 Drive | [계획서 13절](../docs/ml/RETRAINING_EFFECT_EXPERIMENT.md) |
| RF-03 | F4 양방향 오류 교정 | 10-10 | 완료 | 원래 라벨 정확도 test .619 → .278, RouterArena .454 → .244. medium 쏠림 | [보고서](results/retraining-effect-f4-20261010/REPORT.md) |
| RF-04 | F5 라벨 비율을 맞춘 오류 교정 | 10-10 | 완료 | 쏠림 해소(F5−F4 +.30)되었지만 F0 미달(test .580, RouterArena .444) | [보고서](results/retraining-effect-f5-20261010/REPORT.md) |

### 2.4. 평가·분석 (EV)

| 코드 | 실험 | 시기 | 상태 | 핵심 결과 | 기록 |
|---|---|---|---|---|---|
| EV-01 | RouterArena 외부 난이도 라벨 진단 | 09-24~25 | 완료 | 라벨 일치 R0 45.48% / R1 22.80% / R2 21.36%. 항상 easy 47.49% | [평가 기록](results/routing-objectives-continual-v1-A-20260924/routing-diagnostics/run.json), [ML 명세 1.1절](../docs/ml/README.md) |
| EV-02 | R0~R3 종합 판단 | 09-25 | 완료 | 개발 후보 R1-natural, 대안 R3-natural, 즉시 배포 보류 | [리뷰](results/routing-method-review-20260925/REVIEW.md) |
| EV-03 | λ 정밀 탐색 (R0~R3) | 10-02 | 완료 | R0는 λ로 거의 움직이지 않음. R1은 98% 품질 유지에 상대 비용 .39 | [보고서](results/lambda-sweep-20261002/REPORT.md) |
| EV-04 | 라우팅 천장 분석 | 10-13 | 완료 | 학습 라우터는 혼합선 +.02~.04에서 멈춤. 다른 LLM 8개 정답을 알아도 +.052. 라벨 잡음 상한 .794/.875. 상향 허용 시 low부터 시작이 oracle 수준 최종 해결률(v2 93.5% / 88.3%) | [보고서](results/routing-ceiling-20261013/REPORT.md) |
| EV-05 | 통일 평가 시제품 (어댑터·상향 고려 규칙) | 10-13 | 완료 | 어댑터 후 R0 ≈ R1. 상향 고려 규칙 ≈ κ별 최선의 고정 시작 단계 | [README](results/unified-eval-prototype-20261013/README.md) |
| EV-06 | 논문 라우터 비교 | – | 통합(10-16, EV-10) | 비교 대상(kNN 효용 라우터+MLP·SVM, MF, RouterDC, GraphRouter, Hybrid LLM, IRT-Router, RouteLLM-BERT 공개본·재학습본, Avengers-Pro)을 EV-10에서 비교. CARROT은 제외 | [최종 계획서 4절](../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md), [이전 계획서](../docs/ml/ROUTER_COMPARISON_EXPERIMENT.md) |
| EV-07 | 통일 기준 재평가 | – | 통합(10-16, EV-10) | 어댑터·손익분기점 평가 틀은 EV-10의 평가 방식(두 갈래)이 됨 | [최종 계획서 2.5절](../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md) |
| EV-08 | 외부·세대 전이 | – | 통합(10-16, EV-10) | 외부 데이터의 일부를 학습에 쓰기로 해, 전이는 EV-10에서 "A만 학습한 버전 대 섞어 학습한 버전"으로 측정 | [최종 계획서 4절](../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md) |
| EV-09 | RouterArena 해결률 실측 | – | 보류(10-13) | 층화 표본 약 1,000문항을 서비스 후보로 호출. API 비용 때문에 팀원과 협의한 뒤 결정 | [최종 계획서 5절](../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md) |
| EV-10 | 통합 라우터 비교 (모델 집단 단계, 혼합 학습, 두 갈래 평가) | 10-16~ | 진행 중(10-17: 00~06 완료, 07a·07b 다음) | 우리 R0-집단·R1-집단·R1-BGE와 논문 라우터 8종을 A 분할(v2)·LLMRouterBench·SPROUT train으로 학습해 원천 4개 test에서 비교(영어 문항만). 비용은 서비스 가격 1:3:5, 집단 평균 데이터 가격은 민감도. A만 학습한 R1-집단·kNN으로 새 세대 데이터 효과 측정 | [최종 계획서 4절](../docs/ml/ROUTER_FINAL_EXPERIMENT_PLAN.md) |

### 2.5. 캐시 (CA)

| 코드 | 실험 | 시기 | 상태 | 핵심 결과 | 기록 |
|---|---|---|---|---|---|
| CA-01 | 임베딩 계약과 캐시 임계값 (팀원, 8월 선행) | 08 | 완료 | 같은 뜻 최소 .8621 < 함정 최대 .9371이라 코사인 임계값만으로 분리 불가 | [보고서](results/cache-threshold/REPORT.md) |
| CA-02 | 리랭커 4종 선정 (팀원, 8월 선행) | 08-30 | 완료 | quora-distilroberta + BGE AND 조합. 사전 기준은 미통과 | [보고서](results/reranker-20260830/REPORT.md) |
| CA-03 | 리랭커 실적용 검증 (팀원, 8월 선행) | 08 | 완료 | 오탐 19 → 1, 정탐 21 → 19. 운영 enforce 근거 | [보고서](results/rerank-endtoend/REPORT.md) |
| CA-04 | exact match 캐시 (팀원, 8월 선행) | 08 | 완료 | 동일 질문 15,063ms → 5ms | [보고서](results/exact-match/REPORT.md) |
| CA-05 | 캐시 실험 C0~C3 run001 | 10-01~05 | 완료 | exact(C1)는 잘못된 재사용 0. 운영 C3는 함정 질문 72%에 저장된 답 반환. 1% 상한을 만족하는 설정 없음 → exact만 권고. 후속 CA-07~12에서 NLI 판정(C4)으로 바뀌어 운영 배포됨(SY-03) | [계획서](../docs/ml/CACHE_EXPERIMENT.md), [보고서](results/cache-experiment-v1-20261005/REPORT.md) |
| CA-06 | 외부 시맨틱 캐시 도구 비교 | – | 폐기(10-16) | vCache·LangCache 임베딩·GPTCache 판정 모델·NLI 판정기를 운영 C3와 비교하려던 계획. 팀원이 진행한 NLI 판정(C4) 방향으로 가기로 해 병합하지 않았다. NLI 판정기·GPTCache 비교는 CA-09~11에서 이뤄졌다 | PR #54(닫음)의 `docs/ml/CACHE_TOOL_COMPARISON.md` |
| CA-07 | 대화형 300그룹 C3 평가·진단·판정기 비교 (팀원) | 10-10 | 완료(PR #60) | 대화형 질문에서 운영 C3의 함정 통과 12.7%(D2 72%), 같은 뜻 재사용 50.5%(D2 89%). 프로그래밍 분야는 함정 통과 31%가 재사용 20%보다 많음 | [보고서](cache-natural-v1/REPORT.md) |
| CA-08 | D2 재현본 (팀원) | 10-11 | 완료(PR #60) | CA-05의 D2 원문을 구할 수 없어 같은 방식으로 다시 만듦(300그룹, calibration 100 / test 200). 운영 점수와 일치 확인 | [보고서](d2-replica-v1/REPORT.md) |
| CA-09 | NLI 도입 방식 비교 (팀원) | 10-11 | 완료(PR #60) | 길이별 기준(nli-deberta-v3-base)이 두 데이터 모두 함정 통과 5% 이하에서 재사용 최고(대화형 72.7%/4.7%, D2 재현 53.5%/3.0%) | [보고서](nli-adoption-v1/REPORT.md) |
| CA-10 | 차이 문장 NLI·NLI 모델 7종 (팀원) | 10-12 | 완료(PR #60) | tasksource/deberta-base-long-nli + 차이 문장, 기준 0.2815: 대화형 84.1%/4.7%, D2 재현 95.5%/1.0%. 운영 채택(C4) | [보고서](nli-diff-v1/REPORT.md) |
| CA-11 | GPTCache 0.1.44 실제 실행 기준선 (팀원) | 10-12 | 완료(PR #60) | 기본 설정은 함정 통과 82.8%(대화형)·100%(D2 재현). 함정 5% 이하로 맞추면 재사용 0~29.5% | [보고서](gptcache-baseline-v1/REPORT.md) |
| CA-12 | 배포한 NLI 구성(C4)의 동결 기준 재평가 (팀원) | 10-14 | 완료(PR #60) | 서비스형 신규 세트에서 함정 통과 59.5%(C3) → 4.6%(C4), 같은 뜻 재사용 98.0%. LmArena에서는 사전 기준 미충족(2.3% → 2.9%, 라벨 잡음 큼) | [보고서](cache-eval-v2/REPORT.md) |
| CA-13 | GPTCache 구조 대조 (팀원, 8월 선행 문서, 10-16 등록) | 08-30 | 완료 | GPTCache 기본 흐름(ALBERT 임베딩 → SQLite·FAISS 후보 → 거리 판정)과 우리 설계를 대조했다. 운영 리랭커 quora 모델이 GPTCache 크로스인코더 평가기의 기본 모델과 같아 라이브러리는 도입하지 않기로 함. 실제 실행 비교는 CA-11 | [문서](GPTCACHE_COMPARISON.md) |

### 2.6. 시스템·운영 (SY)

| 코드 | 실험 | 시기 | 상태 | 핵심 결과 | 기록 |
|---|---|---|---|---|---|
| SY-01 | R0 운영 1.0 등록 검증 | 09-27 | 완료 | D_gate 3,581건에서 9개 고정 하한 모두 통과 | [보고서](results/r0-production-qualification-20260927/REPORT.md) |
| SY-02 | 직접 승격 E2E 시험 | 09-27 | 완료 | run-03·04 모두 T4 학습 → 1.1 승격 → serving 반영 → 1.0 rollback 성공 | [ML 명세 5.14절](../docs/ml/README.md) |
| SY-03 | NLI 캐시 재검증 운영 배포 (팀원) | 10-14 | 완료(Spring PR #59, serving PR #13) | 코사인 ≥ 0.80 후보 1개를 NLI 점수 ≥ 0.28154424로 재검증. 운영 점수가 오프라인 평가와 일치(최대 차이 4.8e-7), NLI 검증 지연 p50 567ms(짧은 질문) | [CA-12 보고서 0절](cache-eval-v2/REPORT.md) |

### 2.7. 탐색 분석 (XA, 2026-10-09~13 대화)

정식 실험이 아니라 판단 근거를 찾기 위한 분석이다. 스크립트는 세션 임시 폴더에 있었고, 결론 중 일부는 정식 기록으로 옮겼다.

| 코드 | 분석 | 핵심 결과 | 정식 기록으로 옮긴 곳 |
|---|---|---|---|
| XA-01 | 혼합선·과제 조회·다른 LLM 상한 (RouterBench D_dev) | 비용 .5에서 R1(+.026) ≈ 다른 LLM 8개 상한(+.026). 과제 조회가 R1 최대 이득과 비슷 | EV-04(test로 정식화) |
| XA-02 | 학습 데이터 비용 편차 | 로그 비용 분산 중 같은 모델 안의 몫: RouterBench 5%, xRouteBench 19% | 쓰지 않음. 비용을 서비스 가격으로 고정해 질문별 비용 예측(CARROT)을 하지 않기로 함(10-16). 수치는 이 표가 기록 |
| XA-03 | 세대 간 난이도 전이 | 약한 모델 → GPT-4 AUROC .88. 같은 MMLU 문항에서 2023 → 2025 .67. SPROUT 일반 대화에서 상위 모델이 꼭 필요한 질문 3.5% | 없음 |
| XA-04 | 재학습 데이터 구성 시뮬레이션 (TF-IDF) | 문턱 재보정 후 데이터 구성 간 차이 ±.01 이내 | 없음 |
| XA-05 | 오답 반복 학습 시뮬레이션 (TF-IDF) | stream 정확도 .57 → .998, gate .540 그대로. 푼 비율 회귀가 +.007~.009 | 없음(RF-03·04가 ModernBERT로 확인) |
| XA-06 | 시작 단계별 최종 해결률 (stream) | low부터 시작의 우위 8.3%p(v1) → 5.1%p(v2) | EV-04 4절 |
| XA-07 | v1 채점 오류 영향 정리 | 방법 순위 Kendall τ .93~1.0. 라벨 20% 변경(정정 자체 8.4%). R1 학습 정답 4.6~7.8% 변경 | 계획서 5절 |
| XA-08 | RouterArena와 외부 데이터의 같은 문항 | 700문항(LLMRouterBench 663, SPROUT 38) | 계획서 4절 |
| XA-09 | SPROUT 비용 시험 계산 (calibration만) | 데이터 저자 단가 × 모델별 토큰으로 질문당 비용이 최저 대비 1~232배(토큰만 쓰면 1~1.95배). 세 단계 규칙은 Llama 3.2 3B / GPT-4o-mini / GPT-4o를 고르고, 오픈 모델 단가 0.5~2배에도 같다 | [SPROUT 비용 추산](../docs/ml/SPROUT_COST_ESTIMATION.md) 6·7절. 10-16부터는 SPROUT 모델을 가격대 집단으로 나눌 때만 씀 |

## 3. 관계 한눈에

```text
DT-01 ─┬─ RT-04 ── RT-05 ──┬── EV-01, EV-02, EV-03
       │                   ├── RT-07 ── SY-01 ── SY-02
       │                   └── RF-01 ── RF-02 ── RF-03 ── RF-04
       ├─ RT-06 ── EV-02, EV-03
       └─ CA-05 (운영 R0 고정) ── CA-06(계획)
DT-02 ─── EV-04 · EV-05 ── RT-08 · EV-10 (진행 중, EV-06·07·08 통합) ── EV-09(보류)
DT-03 ─── EV-10 (진행 중)
8월 선행(팀원): RT-01~03, CA-01~04, CA-13
팀원 캐시 후속(10-10~14): CA-05 ── CA-07·08 ── CA-09·10·11 ── CA-12 ── SY-03 (C4 운영 배포)
```
