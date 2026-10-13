# R1에서 R0로 전환하는 재학습 실험

실험 코드: **RF-01** · [실험 목록](../../experiments/REGISTRY.md)

작성일: 2026-09-27. 이 문서는 실험의 사전 계획이며 운영 배포 승인 기록이 아니다. 전체 연구·운영 계약은 [ML 단일 문서](README.md)를 따른다. 실제 실행 config는 `r1-to-r0-transfer-v2-final-only`다. 2026-09-27 run `r1-to-r0-transfer-v2-run001`로 학습·최종 평가를 완료했으며, 결과는 [결과 분석](../../experiments/results/r1-to-r0-transfer-v2-final-only-20260927/ANALYSIS.md)에 둔다. 요약: A 선택 고정, B 가설 불지지, 두 군 모두 망각 기준 미달로 배포 후보 없음.

## 1. 목적과 세 비교군

운영에서 얻는 `finalLabel`과 에스컬레이션 기록만으로는 모든 후보의 성공 여부를 알 수 없다. 따라서 기존 R1의 후보별 성공 예측 학습을 그대로 계속하는 대신, **R1에서 배운 질문 표현을 가져와 R0의 단계 분류로 전환한 뒤 상향 교정 라벨로 추가 학습**할 수 있는지 검증한다. 이번 구현은 실제 사용자 만족 자료가 아닌 공개 benchmark 라벨의 `benchmark_upward_proxy`를 사용한다. 실제 운영 효과를 입증하는 실험과 구분하여, 전환·적응·망각의 통제된 비교를 수행한다. 새 분야·다른 출처에 적응시키는 확장은 **미래 연구이며 이번 졸업프로젝트 범위에서 제외**한다. 사용자가 개인 Colab에서 직접 실행하고, 별도 pilot·중간 전체 dev 평가를 생략하여 실행 비용과 반복 작업을 줄인다.

| 비교군 | 시작 모델 | 학습 목표·출력층 | 실험에서 답할 질문 |
|---|---|---|---|
| A · `r0_continued` | 기존 통합 A의 R0 | 기존 R0 출력층 유지, 3-class CE | 이미 R0로 학습한 모델을 계속 개선하는 기준선 |
| B · `r1_to_r0` | 같은 통합 A의 R1 | R1 인코더를 가져오고 3-class 출력층 새 초기화, CE | R1의 표현을 R0로 옮긴 뒤 같은 피드백으로 학습하면 A보다 유리한가? |
| C · `r1_frozen` | B와 정확히 같은 R1 checkpoint | 기존 sigmoid 출력·고정 라우팅 정책 유지, 학습 없음 | 피드백 대리값으로 바꾼 A/B가 R1을 그대로 사용하는 선택보다 나은가? |

CE는 한 질문의 정답 단계 하나를 맞히는 손실이다. R1의 BCE는 후보별 성공 여부를 각각 맞히는 손실이므로 여러 후보가 동시에 성공할 수 있다. 출력 개수가 둘 다 3개라는 이유로 의미를 같게 취급하지 않는다. B는 전환 뒤 **R1에서 초기화한 R0 모델**이며, 후보별 성공 확률을 계속 예측하는 R1 모델이 아니다.

이 세 비교는 실제로 선택할 세 경로를 비교한다. B는 출력층을 새로 만들고 A는 유지하므로, 차이를 순수하게 초기 학습 목적함수 하나의 인과 효과라고 해석하지 않는다. 이를 분리하려면 `R0 인코더 + 새 출력층`이라는 네 번째 보조 실험이 필요하다. 이번 기본 범위에는 포함하지 않는다. C에는 학습 비용이 없으므로 세 군이 같은 총 계산량을 쓴다고 주장하지 않는다.

## 2. 학술적 근거와 검증할 가설

| 근거 | 확인된 사례 | 이번 실험에서 사용하는 범위 |
|---|---|---|
| [GoEmotions, ACL 2020, §5–6](https://aclanthology.org/2020.acl-main.372.pdf) | 다중 감정 예측으로 학습한 BERT의 출력층을 교체하여 다른 감정 분류 과제로 전이. 후속 과제에는 단일 감정 라벨의 ISEAR도 포함 | 출력 의미를 바꾸면서 인코더를 재사용하는 언어모델 선행 사례. 적은 후속 데이터에서 이득이 있었지만 충분한 데이터에서는 차이가 작아짐 |
| [STILTs, §3–4](https://arxiv.org/abs/1811.01088) | 중간 지도학습 후 다른 과제로 추가 학습. 과제별 출력층과 optimizer 재설정 | 단계별 목적함수 변경과 weight-only 초기화의 근거 |
| [ViT-22B, ICML 2023](https://proceedings.mlr.press/v202/dehghani23a.html) | 다중 라벨 sigmoid cross-entropy 사전학습 뒤 일부 후속 실험에서 softmax 분류 학습 | BCE 계열에서 단일 분류 학습으로 바꾸는 것이 가능한 사례. 이미지 분야의 성과를 라우터 성과로 옮겨 주장하지 않음 |
| [Intermediate-Task Transfer Learning, ACL 2020](https://aclanthology.org/2020.acl-main.467/) | 110개 중간·최종 과제 조합에서 양의 전이와 음의 전이 모두 관찰 | R1에서 출발하면 반드시 좋아진다는 전제를 금지 |

주 가설은 “같은 R0 적응 예산에서 B가 A보다 좋은 품질을 유지하면서 상향 교정 대리 신호를 배운다”이다. 별도로 “B가 동결 R1인 C보다 유용한 운영 경로인가”를 검증한다. R1의 기존 성능 이득이 인코더보다 성공 점수·비용 선택 정책에서 비롯되었다면, R0 전환 시 이득이 사라질 수 있다. 현재까지 조사한 선행 연구가 우리의 ModernBERT·후보 풀·에스컬레이션 조건을 그대로 검증한 것은 아니다.

## 3. 디렉터리와 실행 파일

파일은 기존 R0/R1/R2 및 G1 실험과 섞이지 않도록 [별도 노트북 디렉터리](../../ml/notebooks/r1_to_r0_transfer/)에 둔다.

| 파일 | 역할 |
|---|---|
| [`00_prepare.ipynb`](../../ml/notebooks/r1_to_r0_transfer/00_prepare.ipynb) | 환경·Drive 경로·원본 artifact·분할 검증, 공통 실행 계약 준비 |
| [`01_r0_continued.ipynb`](../../ml/notebooks/r1_to_r0_transfer/01_r0_continued.ipynb) | A의 공통 CE 적응·동일 피드백 학습. 중간 전체 dev 평가 없음 |
| [`02_r1_to_r0.ipynb`](../../ml/notebooks/r1_to_r0_transfer/02_r1_to_r0.ipynb) | B의 출력층 전환·공통 CE 적응·동일 피드백 학습 |
| [`03_r1_frozen.ipynb`](../../ml/notebooks/r1_to_r0_transfer/03_r1_frozen.ipynb) | 설명용. C는 04에서 함께 평가하며 별도 실행 불필요 |
| [`04_compare_results.ipynb`](../../ml/notebooks/r1_to_r0_transfer/04_compare_results.ipynb) | A/B 전체 학습 후 baseline/final dev → 선택 동결 → 보호 평가·최종 보고서 |
| [`experiment_config.json`](../../ml/notebooks/r1_to_r0_transfer/experiment_config.json) | 비교군·seed·길이·학습량 등 공통 설정 |
| [`source_pins.example.json`](../../ml/notebooks/r1_to_r0_transfer/source_pins.example.json) | 개발용 pin 형식 예시. 사용자 실행에는 kit의 완성된 source_pins.json 사용 |
| [`experiment.py`](../../ml/notebooks/r1_to_r0_transfer/experiment.py) | 노트북들이 공유하는 검증·실행 보조 코드 |
| [`prepare_local_data.py`](../../ml/notebooks/r1_to_r0_transfer/prepare_local_data.py) | 기존 봉인 artifact에서 6개 Parquet·실제 pin·정책·archive 목록 생성 |
| [`prepare_models.py`](../../ml/notebooks/r1_to_r0_transfer/prepare_models.py) | 00에서 기존 개인 Drive tar를 검증하고 모델 복구 |

사용자에게는 예시 pin이 아닌 실제 원본 artifact에서 자동 고정한 데이터·checksum·경로를 포함한 kit를 제공한다. `user-kit/transfer-experiments`를 개인 Drive의 `MyDrive/LLM_Router/` 아래에 올리고 **00 전체 실행 → 01/02를 별도 GPU 세션에서 전체 실행 → 둘 다 끝나면 04 전체 실행**한다. 03은 설명용이므로 실행할 필요가 없다. 사용자가 수동으로 pin이나 SHA를 작성하지 않는다. 모든 세션은 같은 config·입력을 사용하며 소스 디렉터리에 인증 파일을 넣지 않는다.

실험 출력은 개인 Drive의 `LLM_Router/transfer-experiments/runs/<RUN_ID>/` 아래에 둔다. 실제 구성은 다음과 같다.

```text
shared-data/                         # bridge 및 여섯 cycle의 공통 학습 파일
arms/<arm>/seed-<seed>/<phase>/       # stage 계약, checkpoints/, model/, 완료 기록
scores/<arm>/seed-<seed>/<phase>/     # 역할·길이별 예측과 검증 receipt
selection-lock.json                  # 개발검증 선정 동결
report/                              # CSV, 상태, 그림, REPORT.md
```

각 세션은 자신의 arm/seed 경로만 쓴다. C는 `r1_frozen/seed-0/baseline` 평가 결과만 생성한다. 최종 보고서의 로컬 반입 위치는 `experiments/results/r1-to-r0-transfer-v2-final-only-<실행일>/`로 구분할 수 있지만 자동 반입 기능은 아니다. 기존 실험 결과를 덮어쓰지 않는다. 소스 ZIP은 [`build_bundle.py`](../../ml/notebooks/r1_to_r0_transfer/build_bundle.py)로 생성하여 kit에 포함했고, 노트북은 제공된 SHA 파일을 읽어 자동 검증한다. 자세한 입력·실행법은 [노트북 README](../../ml/notebooks/r1_to_r0_transfer/README.md)를 따른다.

## 4. 출발점의 공정성과 원본 고정

주 비교는 **같은 통합 A 데이터·후보·T1·초기 학습량으로 만든 A R0와 A R1**을 사용한다. 단일 문서 §1.1에 기록된 통합 A의 초기 학습 조건은 같은 21,406문항, seed 42, 최대 길이 512, 2,007 optimizer step이다. 이 숫자는 과거 기록이며 이번 파일의 행 수·checksum 검증을 대신하지 않는다.

G1 R1-natural은 학습 질문·후보 풀·분할이 바뀐 모델이다. 이것을 A R0와 비교하고 차이를 “R1 목적함수의 효과”라고 부르지 않는다. G1 전환 비교나 다른 분야·출처를 추가하는 실험은 이번 졸업프로젝트의 실행 범위에 넣지 않고 미래 연구로 남긴다. 과거 G1 실험 결과를 이번 동일 데이터 비교의 결과로 합치지 않는다.

시작 전 다음 사항을 고정한다.

1. R0/R1 checkpoint 경로·SHA-256·학습 manifest·원본 Git revision.
2. pretrained ModernBERT revision, tokenizer 파일, 입력 정제·라벨·후보 순서.
3. 학습 질문과 평가 질문의 ID·정규화 입력 hash·duplicate group.
4. 각 원천의 실제 low/medium/high 후보 ID와 비용 정의, R1의 기존 선택 정책과 hash.
5. 모든 평가 partition의 원본 hash와 역할, 피드백 생성 규칙 및 trace hash.

두 시작 모델의 데이터·후보·라벨 계약이 맞지 않거나 D_stream/D_gate를 초기 학습에서 이미 보았다면 자동으로 진행하지 않는다. 새로 호환되는 anchor를 만들거나 실험을 별도 비교로 재정의해야 한다. 원천 이름을 입력 질문에 붙여 source를 알려주지 않는다.

## 5. 데이터 역할과 누수 방지

**이번 재학습 데이터의 출처도 RouterBench+xRouteBench다. 새로운 외부 데이터셋을 확보한 실험이 아니다.** 기존 학습 모델을 불러와 사용하고 초기학습을 다시 수행하지 않는다. 초기학습에 사용한 train은 출력층 전환 적응·망각 방지용으로 재사용하고, 처음부터 따로 남겨 둔 feedback 분할을 새로 들어온 질문의 대리 자료로 사용한다. RouterArena는 평가 전용이다. 이번 run용 6개 Parquet·원본 모델 pin·정책·archive 목록은 kit에 준비했다. 공통 feedback 로그는 00 실행 시 원본 R0 추론으로 생성한다. 입력 준비 완료와 GPU 학습 완료를 구분한다.

2026-09-27 원본 저장소의 기존 artifact를 읽기 전용으로 대조했다. `ml/artifacts/routing-objective-continual-v1/dataset/manifest.json` 및 `ml/artifacts/frozen-evaluation-outcomes/continual-v1-20260924-28cbc342c510/manifest.json`에 대응하는 train 21,406행, dev 3,581행, stream 5,368행, gate 3,581행, retention 1,797행이 존재한다. 관련 Parquet의 manifest SHA가 일치하며, train/dev와 stream/gate/retention 사이 sample ID·duplicate group·정규화 입력의 교집합은 0이었다. 초기 Colab `ml/artifacts/routing-objective-continual-v1/input-manifest.json`의 입력 역할은 D_initial_train/D_dev로 기록되어 있다. 이 원본 검증을 토대로 `prepare_local_data.py`가 이번 run용 정규화 데이터와 pin을 생성했다. logger 추론은 Colab 00에서 수행한다. RouterArena full은 8,379행을 유지하며, 그중 240행은 같은 본문을 공유하는 다른 질문이므로 duplicate group 반복을 이유로 제거하지 않는다. 그룹 안의 여러 질문은 유지하고, 역할 간 같은 그룹의 누수는 금지한다.

기존 continual 분할을 재사용하며, 결과에 유리하도록 다시 나누지 않는다. 설계 비중 60/10/10/15/5는 설명용 계약이고 실제 행 수는 준비 결과에 기록한다. 이번 helper는 아래 **여섯 역할만** 받아 각각 `<role>.parquet`로 읽는다. 봉인 RouterBench/xRouteBench test를 별도로 읽는 adapter는 후속 범위다.

| 데이터 | 역할 | 금지 사항 |
|---|---|---|
| `train` · D_initial_train · 60% | A/B의 공통 전환 적응, 이후 base rehearsal | 평가 질문을 보충해서 넣지 않음 |
| `dev` · D_dev · 10% | 모든 학습 완료 뒤 baseline/final 평가와 A/B 최종 선택. trainer 내부 작은 validation 최대 48행의 원천 | 중간 전체 dev 별도 추론·학습량 튜닝·외부 test 기반 재선정 금지 |
| `gate` · D_gate · 10% | 모델·정책 lock 뒤 최종 연구용 자격 검증. 이번 주 실험에서는 중간 경로 선택에 사용하지 않음 | gradient 학습·hyperparameter 탐색·승자 재선정 금지 |
| `feedback` · D_stream · 15% | 여섯 request window와 공통 피드백 대리값 | 아직 도착하지 않은 질문의 라벨을 미리 학습하지 않음 |
| `forget` · D_retention_test · 5% | 최종 기존 영역 성능 유지 확인 | rehearsal·gate·모델 선택에 사용하지 않음 |
| `routerarena` · RouterArena | 최종 외부 인지 난이도 정렬 | 학습·피드백·승격 gate·모델 선택 금지 |

RouterArena는 과거 결과를 이미 관찰했으므로 새로운 blind test라고 쓰지 않는다. 현대 서비스의 Claude 후보 결과가 없는 RouterArena 정확도를 Claude 답변 성공률로 해석하지 않는다. 원천별 후보 풀과 품질 지표가 다르므로 source별 지표를 먼저 보고하고, 원천을 동일 비중으로 평균한 값임을 표시한다.

초기 라벨은 `benchmark_difficulty`, 이번 피드백 대리 라벨은 `benchmark_upward_proxy`로 기록한다. 후속 실제 로그에 쓰는 `observed_escalation_action`과 구분한다. 세 값이 모두 정수 0/1/2로 표시된다는 이유로 같은 정답 증거라고 설명하지 않는다.

## 6. 단계별 실행

### 6.1. 입력 준비 후 직접 학습

사용자는 준비된 kit를 개인 `alexyoo0506@gmail.com` Drive에 업로드하고 00을 전체 실행한다. 노트북이 기존 Drive tar에서 모델을 복구하고 파일 SHA-256·분할 간 누수·후보·모델 출력 의미를 자동 검증한다. 빈 pin 작성이나 추가 checksum 입력은 필요하지 않다. 필수 검증 뒤 공통 feedback logger 추론과 공통 학습 입력 준비를 수행한다.

별도 40-step pilot과 중간 전체 dev 평가를 수행하지 않는다. 공통 trainer의 작은 validation 최대 48행은 내부적으로 유지하지만 사용자가 추가 평가 노트북을 실행하는 단계는 아니다. 학습량·optimizer·seed는 아래 고정 설정으로 실행하며, 결과를 보고 특정 비교군만 학습량을 늘리지 않는다. B의 encoder 복원·head 교체 확인은 모델 변환 검증으로 처리한다. 중단 복구용 25-step checkpoint 저장은 평가 생략과 무관하게 유지한다.

### 6.2. 기준 모델 보존과 최종 일괄 평가

기존 R0와 기존 R1의 원본 모델·정책을 수정하지 않고 보존한다. B가 전환하기 전 C와 같은 R1 checkpoint인지 checksum으로 확인한다. 랜덤 새 출력층, bridge 종료 또는 중간 cycle마다 전체 평가셋을 별도 추론하지 않는다.

A/B의 모든 seed가 cycle-06까지 끝난 뒤 **04 노트북 한 차례의 최종 평가 흐름**을 실행한다. 먼저 원본 baseline과 최종 checkpoint의 D_dev를 8192로 평가하여 A/B 선택을 고정한다. 이어 명시적 최종 평가 플래그로 `gate/forget/routerarena`의 baseline·최종 checkpoint를 평가하고 보고서를 생성한다. 여러 모델·역할에 대한 최종 추론은 필요하지만, 학습 도중 전체 dev 추론을 반복하지 않는다. 외부 결과로 승자를 다시 고르지 않는다.

### 6.3. Phase 1 — 공통 R0 전환 적응

A는 기존 CE 출력층을 유지한다. B는 **ModernBERT 본문 encoder를 복원하고 ModernBERT의 `head`와 `classifier`를 새로 초기화**한다. 두 모델 모두 D_initial_train의 같은 질문·라벨·순서에 대해 같은 실제 optimizer step을 실행한다. B만 추가 head 예열이나 더 많은 epoch를 주지 않는다. C에는 이 학습을 하지 않는다.

기존 optimizer·scheduler는 가져오지 않는다. 새 optimizer의 동일 설정을 쓰고, bridge 종료 checkpoint를 phase 2의 각 CE anchor로 삼는다. 기존 R1용 policy·success metadata를 B의 CE package에 잘못 남기지 않는다. 원본 R1 checkpoint를 수정하는 대신 새 objective·parent model hash를 기록한다.

### 6.4. Phase 2 — 같은 피드백 대리 신호로 여섯 번 추가 학습

이번 최소 실험은 기존 단일 문서 §5.7의 **공통 고정 logger** 원칙을 가져오되, 관측 가능한 benchmark 라벨을 쓰는 `benchmark_upward_proxy`로 범위를 한정한다. 질문별 모든 모델의 응답을 새로 생성하지 않는다. 공개 난이도 라벨을 실제 사용자 만족 또는 실제 후보의 성공 결과로 바꾸어 부르지 않는다.

1. D_stream의 질문을 `stream_seed=20260927`의 stable hash 순서로 정렬하고 여섯 window를 **필터링 전에** 고정한다. 원래 서비스의 시간순 도착을 재현하는 실험이라고 표현하지 않는다.
2. 원본 R0를 동결한 공통 logger로 각 질문의 `initial_label`을 예측한다. benchmark의 기존 난이도 라벨을 `final_label` 대리값으로 사용하고 `final_label > initial_label`인 행만 사용한다.
3. 각 window에서 상향 조건을 통과한 질문 중 stable hash로 **정확히 32행**의 새 피드백을 선택한다. A/B가 각자 피드백을 만드는 on-policy 실험은 이번 범위가 아니다.
4. 어느 window든 유효 피드백이 32행 미만이거나 log·라벨·질문 출처를 검증할 수 없으면 준비 단계에서 실패시킨다. 실행 중 특정 군만 적은 step으로 넘어가거나 가짜 피드백을 채우지 않는다. 새 계약이 필요하면 결과를 보기 전에 사유·버전을 남긴다.
5. 현재 새 피드백 32행에, 이전 window의 history를 **라벨당 최대 32행**, 초기 train rehearsal을 **라벨당 최대 32행** 선택하여 섞는다. strict upward history에는 easy가 없으므로 전체 최대 64행, base는 세 라벨 합계 최대 96행이다. 라벨별 stable hash 표본이며 source별 층화는 하지 않는다. A/B에는 같은 표본·순서를 제공한다. D_dev에서 라벨당 최대 16행, 합계 최대 48행의 작은 validation을 공통 trainer가 내부적으로 유지한다. 이 행은 gradient 학습에 넣지 않으며, 중간 전체 D_dev를 별도 추론하는 실행은 생략한다.
6. A/B는 bridge 이후 자기의 직전 학습 checkpoint에서 시작하여 각 window 100 step씩 진행한다. 모든 고정 예산 후보를 다음 단계로 이어 가며, 중간 gate로 특정 군의 학습 경로를 선택하지 않는다.

피드백 log에는 question/sample ID, initial/final label, logging model hash, feedback 데이터 SHA와 provenance를 남긴다. 생성 보조 명령은 `build-feedback-log --config CONFIG --pins PINS --data-dir DATA --output-file LOG`이며, 준비 명령에는 `--feedback-log LOG`를 명시한다. sidecar 없는 임의 CSV를 검증된 실제 사용자 피드백으로 받아들이지 않는다. 실제 관측한 에스컬레이션 로그로 학습하는 adapter는 후속 범위다. 평가 데이터에 이미 검증된 후보 quality/cost 열이 있으면 선택 품질·비용 수치만 계산할 수 있으며, 없으면 이를 만들어 내지 않는다.

이 proxy가 검증하는 것은 **상향 교정 형태의 약한 라벨을 받아 적응·망각하는 성질**이다. 사용자가 그 단계에 만족했는지, 그 모델이 실제로 성공했는지, 그 단계가 최소 충분 후보인지는 검증하지 못한다. `label_origin=benchmark_upward_proxy`를 원본 `benchmark_difficulty`와 함께 보존한다.

이번 과학적 비교는 운영의 “세 성공마다 anchor 교체”, minor의 cycle anchor 재시작, 중간 승인·거절에 따른 version 변경을 재현하지 않는다. 모든 후보를 순차 진행해 전환 효과를 비교한 뒤 최종 gate를 확인한다. 운영형 gate·anchor·publisher의 통합 검증은 방식 채택 이후 별도 수행하며, 이 실험 성공을 운영 파이프라인 완성으로 표시하지 않는다.

## 7. 공통 학습·평가 조건

| 항목 | 기본 계약 |
|---|---|
| 반복 | adaptation seed 42/43/44 |
| 초기 checkpoint | 기존 seed 42로 만든 A R0/A R1. 세 adaptation seed를 세 번의 독립 초기학습으로 표시하지 않음 |
| A/B 학습 길이 | 512 token. 기존 학습 조건을 유지하고 전이 방식만 우선 비교 |
| 평가 길이 | 8192 token만 사용, 같은 tokenizer·전처리·truncation. 과거 길이 검증을 반복하는 512 진단은 이번 범위 제외 |
| 공통 loss | A/B 모두 3-class CE, 같은 라벨 순서 easy/medium/hard |
| 표본 분포 | bridge는 전체 train의 자연분포. feedback에는 상향 필터, replay에는 라벨별 quota가 있으므로 혼합 전체를 자연분포라고 부르지 않음. A/B 동일 표본과 shuffle, `sample_weight=1`; 별도 balanced sampler·class weight 없음 |
| 학습량 | bridge 200 optimizer step, 각 feedback window 100 step. 완료 step을 실제 확인 |
| optimizer | AdamW, 학습률 `2e-5`, weight decay `.01`, warmup ratio `.1` |
| batch | micro batch 8 × accumulation 4 = effective batch 32. 두 군에 같은 값 적용 |
| C 정책 | 원천별 sigmoid 및 봉인된 비용 선택 정책. 실제 후보 pool이 없는 RouterArena에서는 비용 없는 sigmoid argmax 진단. 외부 평가로 재튜닝하지 않음 |
| A/B 정책 | 같은 CE argmax. 이번 실행에서 새 정책 튜닝은 하지 않음 |

같은 `max_steps`를 쓰는 것만으로 공정성을 주장하지 않는다. 데이터가 적으면 epoch가 먼저 끝날 수 있으므로 충분한 epoch를 설정하고 실제 완료 step을 확인한다. 총 사용 token·truncation 통계는 현재 helper의 자동 수집 항목이 아니므로 별도 계측해야 한다. 실패한 run도 실행 시간과 비용 분석에서 제외하지 않는다. 8192 **학습**은 이번 기본 실험이 아니며, 추론 길이 채택을 학습 길이 변경으로 소급 표시하지 않는다.

C는 학습하지 않기 때문에 동일 입력·정책에서 seed별 예측이 같은 기준선이다. 계산을 절약하려면 한 번 추론한 결과를 검증된 hash로 재사용한다. 이를 독립적으로 학습한 세 R1 모델로 집계하지 않는다.

## 8. 평가표·시각화와 채택 판단

| 평가 축 | 필수 산출물 |
|---|---|
| 단계 선택 | 정확도·macro-F1·hard recall·hard→easy 오류, 혼동행렬 |
| 라우팅 진단 | 원천별 선택 단계·오류. 검증된 `quality_low/medium/high`, `actual_cost_low/medium/high` 열이 있으면 선택 품질·비용 수치 계산. 실제 상향 후 해결률은 이번 helper의 평가가 아님 |
| 피드백 적응 | baseline과 cycle-06의 최종 전후 비교. 실제 학습 step·피드백 수 기록, 중간 성능 곡선 없음 |
| 망각 | 최종 lock 뒤 baseline 대비 `forget`의 source-macro-F1 변화. 별도 봉인 test 평가는 후속 범위 |
| 기존 평가 자료의 진단 | 같은 데이터의 source×category별 표본 수·지표·오류 변화, RouterArena 정렬. 새 분야·출처 적응 실험은 아님 |
| 자원 | run의 실측 시간·실패 이력·메모리를 확인하고 Colab CU는 수동 기록. 전체 token 수와 truncation 자동 집계는 미구현 |

최종 보고서는 baseline과 cycle-06의 비교표·전후 비교 그림, source×category 성능 heatmap, 혼동행렬을 제공한다. bridge·중간 cycle 예측 및 학습 곡선은 만들지 않는다. 품질–비용 열이 있는 경우 수치가 CSV에 들어가지만 별도 산점도는 아직 생성하지 않는다. source-category CSV의 표본 수를 함께 확인하며, heatmap의 빈 셀은 0점으로 넣지 않는다. 작은 표본의 별도 시각적 경고는 보고서 검토 단계에서 추가해야 한다. 이번 목적에 불필요한 medium 비율 중심 분석은 추가하지 않는다.

고정 질문의 차이는 duplicate group 단위 paired bootstrap으로 요약한다. 한 그룹에 여러 질문이 있으면 해당 질문 전체를 함께 재표집하며, 그룹을 임의로 한 행으로 축소하거나 질문을 독립 표본으로 흩어 뽑지 않는다. seed별 값·평균·범위를 함께 보고한다. 세 seed만으로 초기학습의 불확실성 전체를 추정했다고 주장하지 않는다. 시계열 여섯 window를 서로 독립인 여섯 모델처럼 계산하지 않는다. BCE와 CE loss의 숫자를 직접 비교해 승자를 정하지 않는다.

실행 config의 사전 제안 기준은 **B의 source-macro-F1가 A보다 0.01을 초과하여 높으면 B, 그렇지 않으면 A 유지**다. 동점·작은 차이에 더 복잡한 전환을 택하지 않는다. 세 adaptation seed를 집계한 D_dev로 A/B를 선택하고 C는 고정 참조군으로 보고한다. `lock --run-dir RUN`은 모든 A/B seed의 cycle-06 checkpoint·개발검증 예측과 C의 개발검증 예측이 갖춰진 뒤에만 selection lock을 만든다. 최종 보호 데이터 점수로 승자를 바꾸지 않는다. 다음 조건이 충족되어야 B를 유망한 운영 후보로 판단한다.

1. A와 비교해 공통 입력·피드백·학습량에서 품질이 개선되거나, 사전에 정한 비열등 범위 안에서 다른 운영 이점이 확인된다.
2. C와 비교한 최종 결과와 각 원본 baseline 대비 전후 변화를 공개한다. 중간 평가를 생략하므로 전환 직후의 영향과 각 cycle의 기여를 따로 분리했다고 주장하지 않는다. 더 비싼 단계로 보내 얻은 품질 증가와 같은 비용에서의 개선을 구분한다.
3. 기존 영역 회귀·중요 오류·자원 한도가 사전 기준을 만족한다. 초안에서 forget **source-macro-F1**의 초기 대비 감소 허용폭은 0.01이다. `retention-check.csv`는 seed별로 A의 원본 R0, B의 원본 R1 정책을 기준으로 검사한다. B에는 학습 망각뿐 아니라 출력 의미·선택 정책 전환 효과도 포함된다. 이는 검증된 운영 기준이 아니라 사전 제안값이며 config와 일치시켜 실행 전에 동결한다. 보호된 평가가 실패하면 채택을 보류하고 기존 선택을 외부 점수에 맞춰 재튜닝하지 않는다.
4. seed별 방향이 불안정하거나 실제 서비스 후보로의 이전 근거가 부족하면 추가 검증 대상으로 남긴다.

세 비교군의 결과가 좋아도 이 실험만으로 공개 benchmark 후보를 실제 Claude 서비스 후보로 바꿨을 때의 성공률을 보장하지 않는다. 자동 배포나 GCS `CURRENT` 변경은 이 연구의 완료 조건이 아니다.

## 9. 재현·중단 복구·예산

각 run은 최소한 `run_id`, arm, seed, parent checkpoint hash, config hash, split/trace hash, code revision, 실제 학습 길이·평가 길이, 예정/완료 step, 시작/종료 시간, 실패 사유를 기록한다. 원본 데이터 행 수, 중복·제외 이유, 후보 ID, policy hash, 단계별 checkpoint와 예측 파일 checksum도 보존한다.

Drive에는 완료한 phase별 모델과 검증 가능한 완료 기록을 남긴다. 공통 trainer가 **25 optimizer step마다** 모델·optimizer·scheduler·RNG·학습 cursor를 checkpoint에 저장하고 최근 **2개**를 보존한다. 중단 시 같은 phase의 최신 완료 checkpoint에서 재개하며, 완료 phase는 모델 hash 검증 뒤 재사용한다. 새 phase에서는 optimizer를 새로 만들지만 같은 phase 재개에는 저장된 상태를 복원한다. 다른 seed나 새 출력층 초기화를 같은 run의 연속 step처럼 섞지 않는다. 사용자는 `alexyoo0506@gmail.com` Drive의 5TB 용량을 확인하고 용량 확인 절차 생략을 지시했다. 이에 따라 잔여 용량 입력·사전 차단을 제거하고 해당 계정의 Drive에 보관한다. 저장 계정은 Colab·Drive 연결 시 확인하며, 다른 Google 계정으로 대체하지 않는다. 단계별 최종 모델과 복구 checkpoint의 보관 정책은 유지한다.

실험은 사용자가 개인 계정의 Colab Pro에서 직접 실행한다. 노트북 준비나 실행으로 Vertex 작업을 자동 제출하거나 GCP 서비스 크레딧을 사용하지 않는다. 같은 개인 계정·Drive 연결·현재 GPU를 확인하고, 실행 전후 Colab CU 잔액을 수동 기록한다. CU 자동 조회·집계는 구현하지 않았다. 두 GPU 세션을 사용할 수 있으면 A와 B를 병렬 실행할 수 있지만 같은 arm/seed 경로에는 두 세션이 쓰지 않는다. C 평가와 결과 결합은 A/B 학습이 끝난 뒤 04 노트북에서 한 번에 수행한다.

기본 학습 run 수는 A/B × 3 adaptation seed = 6개다. 각 run 안에 bridge 1회와 6회의 피드백 학습이 있다. 전체 완료 기준은 42 stage이며, 사전 검증에서 32행 미만인 window는 실패시킨다. 실패·중단 run을 완료한 학습으로 집계하지 않는다. C는 추가 학습 0회다. 별도 pilot 비용은 쓰지 않고 첫 본 학습 stage의 기록으로 남은 시간을 추정할 수 있다. 이 추정으로 특정 군의 고정 학습량을 임의 변경하지 않는다. 과거 GCP의 회당 $1–3 예산이나 2시간 초기학습 기록을 이 다단계 Colab 실험 전체 비용으로 복사하지 않는다.

## 10. 실행 전·완료 체크리스트

- [x] A R0/A R1의 실제 파일과 SHA-256, 동일 초기 데이터·후보 계약 확인.
- [x] D_initial_train/dev/gate/stream/retention 및 외부 평가의 ID·중복 그룹 누수 검사.
- [x] 고정 benchmark-upward proxy log·sidecar와 label provenance 준비. 여섯 window 각각 유효 표본이 최소 32행인지 검사. 실제 운영 만족 데이터라고 부르지 않음.
- [ ] config와 notebook helper의 실제 지원 범위 확인. 별도 운영형 gate·anchor, 실제 서비스 outcome·상향 해결률 분석을 완료로 표시하지 않음.
- [x] 입력 SHA·누수·모델 전환 계약 확인 후 본 설정 봉인. 별도 pilot·중간 전체 dev 평가 없이 학습 실행. 내부 validation 48행과 checkpoint는 자동 유지.
- [x] 세 seed의 A/B 완료 step 및 동일 feedback 대조. C 모델·정책 불변 확인.
- [x] A/B 전 seed 학습 완료 후 04에서 baseline/final dev 평가 → selection lock → 보호 데이터 최종 평가·보고서 생성.
- [x] 개인 Drive 원본·checkpoint 검증 및 로컬 결과 보고서 보존.
- [x] 연구 결과를 단일 README에 요약. 별도 승인·검증 없는 운영 GCS 게시·서버 변경 없음.

`manifest`는 어떤 파일·설정·출처로 실행했는지를 기록한 목록이고, `split`은 학습·개발검증·최종평가처럼 서로 다른 역할로 나눈 데이터 집합이다. `checkpoint`는 저장한 모델 상태이며, `anchor`는 여러 재학습이 공통으로 출발하는 기준 가중치다. `replay/rehearsal`은 이전 피드백/초기 데이터를 일부 다시 학습해 기존 능력을 유지하려는 방법이다.
