# <run_id> 실험 보고서

<!-- 이 파일을 experiments/results/<run_id>/REPORT.md로 복사한다.
수치의 원본은 같은 폴더 run.json 및 immutable artifact URI에 둔다.
<...>와 미측정 항목을 채우되, 미측정을 0이나 통과로 바꾸지 않는다. -->

## 1. 결론과 상태

<어떤 조건에서 무엇을 관측했는지 한 문장. 검증하지 않은 원인은 가설로 표시.>

| 항목 | 값 |
|---|---|
| run_id / attempt_id | <실험 ID / 실행 시도 ID> |
| 연구 질문 / 비교군 | <검증하려는 질문 / 실제 비교 방법> |
| 실행 범위 | <A 통합 / B source-only / 표현 비교 / feedback / cache; 서로 다른 범위를 합산하지 않음> |
| 실행 | <준비 / 실행 중 / 성공 / 실패> |
| 최종 산출물 보존 | <미확인 / 체크포인트만 검증 / 최종 모델 검증> |
| 방법 선정 / 운영 인수 | <미선정 / validation lock 완료; objective·추론·feedback·gate 호환성 상태> |
| 모델 게시 / 활성화 | <각각 미실행 / 완료; logical·physical version> |
| 근거 수준 | <관측 stdout / 저장된 metrics / 재평가 및 예측 파일> |
| 부모 실행·체크포인트 | <없음 또는 URI + SHA-256 + 재개 step> |

학습 프로세스 성공, 파일 보존, 모델 게시, 서버 활성화는 각각 확인한다.
런타임 종료 전 같은 Drive mount에서 읽힌 파일만으로 클라우드 보존을 확정하지 않는다.
재개한 실행은 같은 seed의 후속 시도이며 독립 반복 실험 수를 늘리지 않는다.
후보의 학습 완료와 전체 비교 완료를 구분한다. R0/R1/R2 중 미실행 방법이 있으면 미실행으로 표시하며 R0만으로 운영 `1.0`을 확정하지 않는다.

## 2. 데이터와 라벨

- 원본: <이름, URL, revision, license 기록 위치, 원본 파일 SHA-256>
- 데이터 계약: <dataset / preprocessing / label rule / split version>
- 누수·제외: <group key, source 간 중복, 제외 건수와 사유, audit URI>
- 라벨: <정확한 경계 부등호, 임계값 선정 데이터와 방법, 동결 시점>
- 후보 pool·성공률: <평가 대상 모델, 유효 분모, 결측 처리; 해당할 때>
- 공통 비교 행: <방법별 sample ID·입력 hash·outcome 결측 제외의 일치 검증; 서로 다르면 공정 비교로 합치지 않음>
- manifest: <immutable URI, SHA-256>

| partition | source별 행 수 | easy / medium / hard | 허용 역할 | 실제 사용 |
|---|---:|---|---|---|
| D_initial_train | <...> | <...> | 학습·train-only 라벨 임계값 선정 | <...> |
| D_dev | <...> | <...> | 개발 검증·사전 선택 | <...> |
| D_gate | <...> | <...> | 반복 승격 판단 | <...> |
| D_stream | <...> | <...> | feedback arrival replay | <...> |
| D_retention_test | <...> | <...> | 망각 최종 진단 | <...> |
| RouterBench / xRouteBench test | <각각> | <각각> | 최종 held-out 평가 | <...> |
| RouterArena | <이번 audit 이후 실제 행> | <...> | 독립 외부 라벨 정렬 | <...> |

D_stream은 검증 데이터가 아니다. xRouteBench 공식 test는 같은 데이터 원천의 held-out이며 별도 외부 benchmark로 세지 않는다. RouterArena의 sub/robustness는 full과 겹치므로 표본 수를 합산하지 않는다.

## 3. 모델·학습 조건과 재현

| 항목 | 고정 설정 / 실제 실행 |
|---|---|
| 모델·tokenizer | <ID, 각각 revision> |
| 입력 / truncation | <조립 계약, max_length, 잘린 행 수/분모, token 분포> |
| objective / sample weight / sampler | <...> |
| 출력·정책 계약 | <class probability / candidate success / relative score, action 순서, policy version·λ·budget> |
| optimizer / lr / decay / warmup / clip | <...> |
| batch / accumulation / effective batch | <...> |
| epoch / planned step / actual step / seed | <...> |
| checkpoint 선택 규칙 | <실행 전 고정한 규칙; 사후 최고 epoch 선택 금지> |
| code | <commit, dirty 여부, snapshot/archive SHA-256> |
| runtime | <Python, torch, transformers, CUDA, GPU, precision, attention, lock URI> |
| 시간·사용량 | <학습/전송/총시간, Colab CU 또는 실제 비용; 미측정은 미측정> |

재현 명령: <고정 config와 데이터·코드 artifact를 이용한 명령>

## 4. 결과

학습 목표 비교는 아래 표에 source별 비용–품질 nAUC와 source macro를 기록한다. 난이도 분류 표를 방법 채택 순위로 대신 사용하지 않는다.

| method / seed | 실제 step | 상태 / 최종 weight SHA | RB D_dev nAUC | xRB D_dev nAUC | source macro nAUC | 선택 정책·검증 비용 |
|---|---:|---|---:|---:|---:|---|
| R0 / <seed> | <...> | <...> | <...> | <...> | <...> | <...> |
| R1 / <seed> | <...> | <...> | <...> | <...> | <...> | <...> |
| R2 / <seed> | <...> | <...> | <...> | <...> | <...> | <...> |

validation lock: <선정 split SHA, 공통 candidate pool hash, source별 normalization·nAUC 정의, tie-break, method 및 policy version, lock SHA>.
불가능한 normalization은 N/A로 남기고 source를 조용히 제외해 macro 평균을 계산하지 않는다. 단일-source B는 해당 source 열만 사용하며 A와 같은 후보 pool의 결과로 합치지 않는다.

| partition / source | n | accuracy | macro-F1 | balanced accuracy | hard recall | hard→easy |
|---|---:|---:|---:|---:|---:|---:|
| <...> | <...> | <...> | <...> | <...> | <...> | <...> |

- 원본 증거: <metrics, epoch history, 예측 Parquet, confusion matrix URI + checksum>
- 학습 곡선: <epoch별 train/dev loss·metric, 필요한 경우 그림>
- 비교·불확실성: <같은 split·조건의 baseline, seed별 결과, group bootstrap CI 또는 미측정>
- outcome 품질·비용·latency: <직접 측정했을 때만 표기; 측정 단위와 분모 포함>
- 선정 상태: <세 방법 비교 완료 여부, 선택 방법, 동률 처리, seed 범위; 미선정이면 null>

라벨 분류 정확도, 실제 선택한 LLM의 답변 품질, RouterArena 인지 라벨 정렬을 구분한다. 라벨 정확도만으로 Claude 정답률·비용 절감·공식 Arena Score를 주장하지 않는다.

## 5. 분석과 다음 결정

1. 관측: <수치와 연결되는 사실>
2. 해석·가설: <관측으로 지지되는 범위와 아직 확인하지 않은 원인>
3. 한계: <단일 seed, 표본·누수·도메인, 미측정, 실패·소실된 파일>
4. 다음 행동: <유지/재실험/중단/승격, 사전 조건; test를 보고 튜닝하지 않음>
5. 운영 인수: <선정 objective에 맞는 inference·confidence·서비스 mapping·관측 가능한 feedback target/loss·gate 변경과 검증 근거; 미지원은 미지원>

## 6. 저장·게시 확인

| artifact | immutable URI / generation | 크기 | SHA-256 / 원격 checksum | 확인 방법·시간 |
|---|---|---:|---|---|
| input bundle | <...> | <...> | <...> | <...> |
| 복구 checkpoint | <...> | <...> | <...> | <...> |
| final model | <...> | <...> | <...> | <...> |
| metrics·predictions·report | <...> | <...> | <...> | <...> |

보관 위치는 Git(작은 config·run.json·보고서), Drive(실험 보관), GCS(검증된 데이터·모델·백업)를 연결한다. 토큰, OAuth code, resumable upload URL은 어떤 보고서에도 넣지 않는다.
최종 게시에는 package 검증 결과와 버전을, 활성화에는 CURRENT generation·서버 desired/serving 일치·실제 추론 확인 결과를 따로 기록한다.

`run.json`은 `schema_version`, `run_id`, `comparison_scope`, `status`, `data`, `training_config`, `code`, `runtime`, `attempts`, `verified_artifacts`, `interpretation`, `adoption_policy`를 포함한다. 목적함수 비교에는 `adoption_policy.required_methods`, `method_status`, `selection_split`, `selection_metric`, `selected_method`, `selection_lock_uri`, `production_activation_allowed`를 기록한다. 숫자 미측정과 미선정은 `null`이며 0·빈 승자 이름으로 대체하지 않는다. execution/artifact/selection/publication/activation의 상태는 각각 증거를 갖는다.
