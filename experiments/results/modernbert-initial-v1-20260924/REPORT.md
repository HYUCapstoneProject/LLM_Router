# ModernBERT R0 후보 학습·복구 — RouterBench + xRouteBench

실험 코드: **RT-04** · [실험 목록](../../REGISTRY.md)

최초 실행은 고정된 3 epoch 학습과 개발 검증을 끝냈지만, 학교 Drive 용량 부족으로 최종 모델 파일이 보존되지 않았다. **최초 실행의 아래 수치는 stdout 관측 결과이며, 게시 가능한 최종 모델의 검증 완료 결과가 아니다.** 모든 파일을 검증한 step 500 체크포인트에서 같은 실험을 복구했으며, 복구 모델의 GCS·개인 Drive 보존까지 검증했다.

기준일: 2026-09-24. 기계 판독 기록은 [run.json](run.json), 공통 형식은 [실험 보고서 템플릿](../../templates/RUN_REPORT.md), 설계 근거는 [ML 최종 명세](../../../docs/ml/README.md)다.

## 1. 목적과 상태

- 목적: 최종 데이터·라벨·512-token 조건의 R0(3-class CE) 후보를 학습·복구해 통합 A의 R0/R1/R2 공통 비교에 연결한다. 운영 `1.0` anchor는 비교 후 선정한 방법으로 확정한다.
- 비교 범위: 초기 R0 한 방법, seed 42. R1/R2와의 통합 A 비교는 완료했으며 [공통 비교 보고서](../routing-objectives-continual-v1-A-20260924/REPORT.md)에 별도로 기록한다. RouterDC, feedback F0/F1/F2 우열 비교는 아직 실행하지 않았다.
- **운영 채택 보류:** 사용자가 R0/R1/R2 중 검증 성능이 가장 좋은 방식을 채택하도록 명시했다. 이 R0 run은 비교 후보·복구 가능한 연구 산출물이며 단독 결과로 정식 운영 1.0을 확정하지 않는다. 통합 A의 동일 조건 세 방법을 D_dev의 source-macro 비용–품질 nAUC로 비교·동결하고, 선택한 방식의 학습·추론·feedback·gate 호환성을 확인한 뒤 게시/활성화한다. 후보별 GCS 연구 백업은 운영 승인과 구분한다.
- 최초 시도: 2,007/2,007 step, exit 0, 모델 reload smoke 관측 성공. 최종 파일 보존 실패.
- 복구 근거: step 500의 model·optimizer·RNG state를 포함한 manifest 전체 파일 SHA-256 검증 및 GCS 백업 완료.
- 모델 GCS 게시·CURRENT 변경·FastAPI 1.0 활성화: 미완료. R0 복구가 끝나더라도 세 방법 비교·선정·운영 인수 전에는 완료로 표시하지 않는다. 현재 운영은 8월 RouterBench-only·8192-token 모델이다.
- 복구 시도는 동일 seed의 재개이며 두 번째 독립 실험으로 집계하지 않는다. 재개 후 실제 지표와 보존 확인 결과를 별도로 추가한다.
- 첫 복구 시도는 난수 상태를 GPU로 역직렬화해 `torch.set_rng_state`에서 중단됐다. `eb016ac8183b1f4ef90a752153476a17898b3ed5`에서 CPU로 읽도록 수정했으며 실제 T4에서 dropout을 포함한 중단·재개와 연속 학습의 가중치가 정확히 일치하는 회귀 검사를 통과했다. 학습 데이터·설정·fingerprint는 변경하지 않았다.

## 2. 데이터·라벨

| 항목 | 값 |
|---|---|
| dataset | `routerbench-xroutebench-continual-v1` |
| preprocessing | `integrated-routing-preprocess-v2-structure` |
| 입력 계약 | `structure-preserving-v2` |
| label rule | `overall-solve-rate-grid-v1` |
| split | `continual-60-10-10-15-5-v1-seed-42` |
| T1 | hard: solve rate < 0.27; medium: 0.27 ≤ rate < 0.64; easy: rate ≥ 0.64 |
| 선정 방법 | D_initial_train만 사용, 9개 후보·group 5-fold OOF·bootstrap 1,000회 |
| T1 선정 통계 | J = 0.577184, bootstrap agreement = 0.850811 |
| T2 민감도 조건 | 0.27 / 0.55, T1 대비 라벨 불일치 10.3102%; 이번 모델 학습에 사용하지 않음 |
| sample weight / sampler | effective weight 1 / 자연 분포 shuffle |

전체 42,935행(RouterBench 34,722 / xRouteBench 8,213)을 보존했다. RouterArena와의 입력 겹침으로 RouterBench 32행을 제외한 뒤 통합 중복 15행을 제거했다. 코드·수식의 기호 및 필수 system/context 구조를 보존하며, source-task ID 재사용을 동일 질문으로 간주하지 않는다. 자세한 원본 URL·revision·SHA·제외 사유는 immutable dataset manifest, experiment-config, contamination-report 및 input-audit에 있다.

| partition | 행 수 | easy / medium / hard | 용도 | 이번 R0 후보 평가 상태 |
|---|---:|---|---|---|
| D_initial_train | 21,406 | 10,608 / 8,121 / 2,677 | 초기 학습·train-only threshold 선정 | 학습에 사용 |
| D_dev | 3,581 | 1,768 / 1,360 / 453 | 개발 검증 | 복구 metrics·공통 routing 평가 검증 완료 |
| D_gate | 3,581 | 1,767 / 1,363 / 451 | 반복 승격 판단 | 준비 완료·미측정 |
| D_stream | 5,368 | 2,651 / 2,040 / 677 | feedback arrival replay | 준비 완료·미사용; 검증셋 아님 |
| D_retention_test | 1,797 | 882 / 685 / 230 | 망각 최종 평가 | 준비 완료·미측정 |
| RouterBench fixed test | 3,473 | 1,669 / 1,412 / 392 | 최종 held-out 평가 | 통합 세 방법의 봉인 test 평가 완료 |
| xRouteBench official test | 3,729 | 2,156 / 829 / 744 | 최종 held-out 평가 | 통합 세 방법의 봉인 test 평가 완료 |
| RouterArena | 원본 full 8,400 | 새 audit 후 확정 | 독립 외부 난이도 라벨 정렬 | 최종 512-token adapter·slice audit 및 평가 미완료 |

60/10/10/15/5는 원본 test를 보호한 나머지 development pool의 분할이다. 내부 평가란 공개 RouterBench+xRouteBench에서 분리한 held-out을 뜻하며 별도의 사내 운영 데이터셋을 뜻하지 않는다. `external_test`라는 내부 split 이름은 xRouteBench 공식 test를 가리킨다. 독립적인 외부 benchmark 원천은 현재 RouterArena 하나다.

RouterArena의 sub_10 809행과 robustness 420행은 full과 겹치므로 독립 표본으로 합산하지 않는다. 누수 감사에 사용한 기존 clean 8,375행을 이번 최종 평가 행 수로 재사용하지 않는다. 명세의 블라인드 인지 감사 300문항·사람 독립 주석 60문항은 실행 전 규모를 동결할 제안이며 아직 수집·평가하지 않았다. 기존 자체 질문 35건, 캐시 질문 92건과 리랭킹 58쌍은 과거 진단 자료다.

## 3. 고정 모델·실행 조건

| 항목 | 설정 |
|---|---|
| 모델 | `answerdotai/ModernBERT-base` |
| revision | `8949b909ec900327062f0ebf497f51aef5e6f0c8` |
| objective / label ID | 3-class CE / easy=0, medium=1, hard=2 |
| max length / precision / attention | 512 / fp16 / eager |
| batch / accumulation / effective batch | 8 / 4 / 32 |
| eval batch / epoch / planned steps / seed | 16 / 3 / 2,007 / 42 |
| optimizer | AdamW, lr 2e-5, weight decay 0.01, warmup ratio 0.1, clip norm 1 |
| gradient checkpointing | 사용 |
| checkpoint 선택 | 사전 고정 3 epoch 종료; 사후 최고 dev epoch 재선택하지 않음 |
| code commit / dirty | `2cee27ef3e7556e5d06dd09626e6f25a31983e42` / false |
| 최초 runtime | Python 3.12.14, torch 2.5.1+cu124, transformers 4.48.3, T4 15GB |
| 최초 실행 시간 | 학습 5,665.158초, 전체 5,750.7573초 (전송 복구 시간 제외) |
| 최초 token 관측 | train+dev 24,987행 중 95행 truncation; 길이 중앙값 79, 최소 10, 최대 986 |
| 사용량·금액 | 이 실행에 귀속한 Colab CU·실제 금액 미측정 |

[고정 학습 config](../../../ml/config/modernbert_initial_v1_training.json), [런타임 계약](../../../ml/config/modernbert_runtime_cu124_v1.json), [실행 진입점](../../../ml/scripts/run-modernbert-initial-v1.py)을 사용했다. 코드 archive에는 train/dev와 manifest만 포함하며 gate/test/stream/retention 질문은 포함하지 않는다.

## 4. 최초 실행에서 관측한 결과

아래는 최종 모델 파일을 재평가한 결과가 아니다. 복구 시도의 저장된 metrics·예측이 확보되면 별도 attempt로 추가하고 최초 관측치를 덮어쓰지 않는다.

| epoch | optimizer step | train loss | dev loss | dev accuracy | dev macro-F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 669 | 0.893819 | 0.863602 | 0.609327 | 0.491640 |
| 2 | 1,338 | 0.806661 | 0.834429 | 0.609048 | 0.517879 |
| 3 | 2,007 | 0.709571 | 0.873627 | 0.605138 | 0.516110 |

| D_dev 라벨 | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| easy | 1,768 | 0.641294 | 0.795814 | 0.710247 |
| medium | 1,360 | 0.556789 | 0.479412 | 0.515211 |
| hard | 453 | 0.500000 | 0.238411 | 0.322870 |

최종 epoch의 dev 정확도는 60.51%, macro-F1은 0.5161이었다. hard recall은 23.84%로 낮다. hard→easy 비율·source/domain별 confusion matrix·독립 test·RouterArena·실제 LLM 답변 품질·비용·latency는 이 실행에서 미측정이다. 예측 파일 없이 이를 역산하거나 만들어 채우지 않는다.

### 복구 실행의 저장된 지표

`eb016ac` 복구는 2,007/2,007 step, exit 0으로 종료했다. 최종 로컬 artifact manifest와 다운로드한 metrics.json의 SHA가 일치한다. 개발 검증 accuracy **0.604580**, macro-F1 **0.518296**, hard recall **0.247241**, loss **0.860558**이다. 최초 stdout 수치와 같다고 가정하지 않으며 이 복구 시도의 값으로 기록한다. 원래 seed 42 checkpoint의 재개이므로 독립 seed 반복은 아니다. 이 분류 지표로 R1/R2의 우열을 결정하지 않는다. 최종 model archive의 GCS checksum과 개인 Drive flush·재연결 후 SHA를 확인했다. D_dev routing nAUC는 0.166115이며, 공통 비교에서 R1이 선정됐다.

## 5. 분석과 다음 결정

1. train loss는 계속 감소했지만 dev loss는 epoch 2 이후 증가했다. 과적합 가능성과 일치하는 관측이며, 단일 seed로 원인이나 최적 epoch를 확정할 수 없다. 이미 정한 3 epoch 예산을 유지한다.
2. hard recall이 easy/medium보다 낮다. 학습의 hard 비중은 약 12.5%, RouterBench 비중은 약 87.7%이지만 분포 차이가 원인이라고 단정하지 않는다. 보존된 모델의 source/domain별 지표와 동일 split baseline이 필요하다.
3. 이 실험은 성공률 기반 3-class 라벨 예측이다. 수치로 실제 Claude 답변 정확도·비용 절감이나 공식 Arena Score를 주장하지 않는다.
4. 순서: 같은 데이터·설정으로 step 500 복구·GCS checksum 검증 → 공통 행·초기화·학습 예산을 확인한 R0/R1/R2 비교 → D_dev source-macro 비용–품질 nAUC로 방법·정책 동결 → 선택한 방법의 추론·feedback·gate 인수 및 package 검증 → 운영 1.0 게시·활성화. pipeline 상태 관측과 공통 데이터 계약 수정은 병행하되 R0 복구만으로 운영 채택을 완료하지 않는다. 독립 test를 반복 선택·승격 gate로 사용하지 않는다.

### 비교 및 채택 상태

| 방법 | 학습 상태 | D_dev 후보 선택 nAUC | 운영 채택 |
|---|---|---|---|
| R0 CE | 2,007 step·GCS/Drive 보존 확인 | 0.166115 | 미선정 |
| R1 success BCE | 2,007 step·GCS/Drive 보존 확인 | 0.863179 | 개발 검증 승자·운영 인수 대기 |
| R2 pairwise | 2,007 step·GCS/Drive 보존 확인 | 0.833874 | 미선정 |

위 표의 실행 범위는 A의 통합 continual split·seed 42다. R1/R2의 후보별 target 결합 후 학습 21,406행·개발 검증 3,581행의 sample ID·입력·순서가 R0와 같고 결측 target이 없음을 검증했다. 실제 학습 step 일치는 각 full run 종료 시 검사한다. 각 방법은 같은 pretrained ModernBERT에서 시작한다. B의 xRouteBench source-only R0/R1/R2+RouterDC 비교와 상위 두 방법의 seed 17/73 반복은 별도 연구 범위이며 아직 실행한 것으로 집계하지 않는다.

세 방법 모두 최종 action을 easy/medium/hard의 3단계와 Haiku/Sonnet/Opus 논리 모델로 연결할 수 있다. 다만 현재 FastAPI의 softmax·난이도 confidence와 finalLabel CE 재학습은 R1/R2의 점수·target 의미에 맞게 변경해야 한다. 선택 방법의 score 변환·서비스 후보 mapping·관측된 feedback target/loss·D_gate 품질/비용 평가를 [운영 인수 계약](../../../docs/ml/README.md#retraining-objective-handoff)에 맞춰 검증한다. 미호출 후보의 outcome이나 사용자 만족에서 객관적 quality 차이를 만들어 채우지 않는다.

공통 비교 코드는 `85c422c9af4c51a51ba5eb45a0ec957b836e02d2`, 입력 bundle SHA-256은 `7ce814776dd9c4875bcc4907a880f3c13411a36f96d9c1667f083219f472fbe0`이다. GPU 실행은 R0 평가 → R1/R2 각각 40-step pilot → 각각 2,007-step full run → 세 방법 선택 lock 순서로 완료했다. 세 방법의 full 학습과 선택 lock 및 원격 보존까지 완료했다. 이후 독립 test 7,202문항도 고정 정책으로 평가했으며 [별도 test 보고서](../routing-objectives-continual-v1-A-20260924/sealed-tests/REPORT.md)에 기록했다. 실제 서비스 인수 완료를 뜻하지 않는다. 보고서는 보존된 지표·예측·lock으로 [생성기](../../../ml/scripts/render-routing-objective-report.py)가 공통 형식과 그림을 만든다.

## 6. 저장 사고와 복구 증거

학교 Drive는 5GB 중 4.71GB가 사용된 상태였다. 최초 런타임에서는 파일이 읽혔지만 새 런타임으로 다시 연결하자 final-model·full.log·full-exit가 없었다. step 1,000은 optimizer state, step 1,500은 weights와 optimizer state가 누락됐다. step 500만 manifest 전체가 통과했다. 따라서 이전의 ‘최종 파일 보존 완료’ 표현은 철회한다.

| artifact | 위치 / 검증 |
|---|---|
| dataset | `gs://llm-router-mlserver/datasets/routerbench-xroutebench/continual-v1-20260924-f6a4cff1684b/dataset/` |
| dataset manifest SHA-256 | `f6a4cff1684bc2626a8901dbfda9f81e7d5d7e8753e5f48320cb1ada44c8cca4` |
| train+dev Parquet SHA-256 | `8e382ee84419aa881ea0c4a68fabb0103945ab3452cdbc98720c149e36eb09ab` |
| input archive SHA-256 | `f235696e6661d19225afab973322a831ca23dd7ee7b906a83f95d7ac584d7d9e` |
| GPU 재개 수정 archive SHA-256 | `7f1ce642921b86feba5f0f6d043ebba924c290c13fc8a315ec2054ee33b5a4d8` (commit `eb016ac`, dirty=false) |
| recovery step 500 | `gs://llm-router-mlserver/training-runs/initial-v1-20260924-2cee27e/recovery-step500.tar` |
| recovery size / generation / CRC32C | 1,814,169,600 bytes / `1790220566576829` / `dNxk0A==` |
| recovery SHA-256 | `d1373c1737a61d7b72fac08d88993701b76553bf11fabac4a209c3314bbe2f40` |
| 검증 방식 | 업로드 완료 응답과 독립 GCS metadata 조회의 크기·CRC32C 일치, 다운로드 SHA-256 일치 |
| 복구 step 1,000 추가 백업 | `checkpoint-step1000.tar`, 1,799,096,320 bytes; SHA-256 `76a18220c577dcb4e29d957abcdb6c544931fbb5275122c0ac8134f6caa62cfb`, CRC32C `F4/ZTQ==`, generation `1790227744090851`; 업로드 응답 크기·CRC 확인 |
| 복구 최종 모델 | `final-model.tar`, 602,142,720 bytes; SHA-256 `d15da837971f5535f1f96dcbe036717d066e339afad923d449c8a74e5a041e4f`, CRC32C `8Cyz2w==`, generation `1790231390158680`; 독립 GCS metadata 크기·CRC·세대 확인 |
| 개인 Drive | 기존 권한으로 연결 완료. `LLM_Router/modernbert-initial-v1-20260924/attempt-recovery-eb016ac`의 복구 입력·checkpoint 4개·최종 모델·로그·종료 기록을 flush·새 mount 이후 크기와 SHA로 검증 완료 |

업로드용 임시 URL·토큰은 Git·보고서·Drive에 저장하지 않는다. 개인 계정에 GCP 프로젝트 IAM을 부여하지 않고 지정된 신규 GCS 객체에만 업로드하는 세션을 사용한다.
