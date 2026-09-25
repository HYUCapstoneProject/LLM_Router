# 봉인 고정 test 평가 보고서

**D_dev에서 선정한 R1을 그대로 유지한다.** 고정 test에서 기본 상대 비용 budget 0.50의 R1 source-macro 평균 품질은 0.789141로 R0의 0.656886보다 높았다. 그러나 비용도 증가했으며 **xRouteBench에서 R1·R2 모두 0.50 비용 상한을 초과했다.** 이 결과는 연구용 test 평가이고 D_gate 통과 또는 운영 활성화 판정이 아니다.

## 1. 평가 범위와 무결성

- 고정 test: RouterBench 3,473행 + xRouteBench 공식 test 3,729행 = 7,202행. duplicate group도 7,202개다.
- 학습은 canonical seed 42 한 번이다. 모든 모델·source별 λ·budget은 기존 validation lock과 일치하며 test를 보고 변경하지 않았다.
- D_dev 선정 지표는 nAUC였지만 이 test에서는 동결한 정책의 실제 품질·비용을 그대로 보고한다. test frontier/nAUC를 만들거나 winner를 다시 고르지 않았다.
- protocol SHA, inference/evaluation manifest SHA와 _SUCCESS, 6개 score Parquet 및 evaluation 산출물의 크기·SHA 검증을 완료했다.
- score 21,606행, method×budget별 예측 129,636행, matched-random 집계 5,400행을 확인했다. 세 방법의 동일 test ID·outcome을 대조하고 36개 method×budget×source 정책의 행동과 평균을 재계산했다.
- model weights와 원천 outcome projection 파일은 이 결과 archive에 없다. 이 작업에서 모델을 로드하거나 원천 dataset을 다시 검증하지 않았다. CI 경계는 해시 검증한 원본 summary에서 읽었으며 bootstrap을 다시 실행하지 않았다.

Protocol SHA-256: `133c093cd569e4d699439585037aa28e8843365e0297aedf8f2c797833114f6b`
원본 summary SHA-256: `29108d02c4f37967ce1bcc69351b56c54cc9df98fcbc601c2b0cd4d5546f254f`

## 2. 기본 budget 0.50의 실제 결과

| source | 방법 | λ | 평균 기록 품질 | 상대 실제 비용 | 0.50 상한 초과 |
|---|---|---:|---:|---:|---|
| routerbench | R0 CE | 0 | 0.589634 | 0.067840 | 아니오 |
| routerbench | R1 success BCE | 0.1 | 0.787100 | 0.489174 | 아니오 |
| routerbench | R2 pairwise | 1 | 0.735531 | 0.140463 | 아니오 |
| xroutebench | R0 CE | 0.003 | 0.724138 | 0.156905 | 아니오 |
| xroutebench | R1 success BCE | 0.03 | 0.791182 | 0.537229 | 예 |
| xroutebench | R2 pairwise | 0.1 | 0.788500 | 0.541151 | 예 |

| 방법 | source-macro 평균 품질 | source-macro 상대 실제 비용 |
|---|---:|---:|
| R0 CE | 0.656886 | 0.112373 |
| R1 success BCE | 0.789141 | 0.513201 |
| R2 pairwise | 0.762016 | 0.340807 |

상대 실제 비용은 source마다 `(정책 평균 비용 − always-low 평균 비용) / (always-high 평균 비용 − always-low 평균 비용)`이다. source macro는 두 source에 같은 비중을 준다. 원천 비용 단위를 합친 달러 비용이나 Claude 절감률로 해석하지 않는다.

- R1은 R0보다 품질과 비용을 모두 높였다. 같은 budget 상한 아래에 고정한 정책이지만 실제 지출이 같은 비교는 아니다.
- xRouteBench의 R1 상대 실제 비용은 0.537229, R2는 0.541151이다. validation에서 찾은 budget 정책이 새 test 분포에서도 비용 상한을 반드시 지킨다는 보장은 없었다. 정책을 test에 맞춰 수정해 초과를 숨기지 않았다.
- R1과 R2의 test source-macro 품질 차이 +0.027126은 관측된 차이다. 이번 bootstrap은 R1-vs-R0와 R2-vs-R0만 포함하므로 **R1-vs-R2 신뢰구간이나 통계적 우월성을 주장하지 않는다.**
- low/medium/high 세 행동은 모든 방법에서 유지된다. R1의 기본 budget 실제 선택 건수는 아래와 같다. 서비스 호환 투영은 low→easy→haiku, medium→medium→sonnet, high→hard→opus이며 이번 실제 후보가 Claude라는 뜻은 아니다.

| R1 source | low | medium | high |
|---|---:|---:|---:|
| routerbench | 16 | 1,945 | 1,512 |
| xroutebench | 699 | 1,308 | 1,722 |

## 3. R0 대비 paired 95% 신뢰구간

아래는 기본 budget 0.50의 **고정된 후보 정책 − 고정된 R0 정책** 차이다. 품질 차이와 비용 차이를 구분한다. 양의 비용 차이는 더 많은 비용을 사용했다는 뜻이다.

| 비교 | source-macro 품질 차이 [95% CI] | source-macro 상대 비용 차이 [95% CI] |
|---|---|---|
| R1 success BCE − R0 | +0.132255 [+0.123020, +0.141503] | +0.400829 [+0.387349, +0.414046] |
| R2 pairwise − R0 | +0.105129 [+0.096476, +0.113759] | +0.228434 [+0.216786, +0.239929] |

| 비교 | source | 품질 차이 [95% CI] |
|---|---|---|
| R1 success BCE − R0 | routerbench | +0.197466 [+0.182738, +0.212439] |
| R1 success BCE − R0 | xroutebench | +0.067044 [+0.056144, +0.077925] |
| R2 pairwise − R0 | routerbench | +0.145897 [+0.131960, +0.159963] |
| R2 pairwise − R0 | xroutebench | +0.064362 [+0.054306, +0.074452] |

Bootstrap은 source와 group의 전체 난이도 label signature로 층화하여 duplicate_group_id 단위로 복원 추출한 10,000회, seed 42다. 동일한 group 추출을 모든 paired 방법·budget·지표에 함께 적용했다. source-macro 비용은 각 source의 high−low 비용 차이로 정규화하고 그 분모도 같은 추출 안에서 다시 계산했다. 비용 정규화의 무효 추출은 0회였다.

이 95% percentile 구간은 **학습된 모델·고정 정책을 조건으로 한 질문 group 표본 변동**을 나타낸다. 모델을 10,000번 다시 학습한 결과가 아니며 학습 seed 변동, 정책 선택 변동, 미래 서비스 분포 변동을 포함하지 않는다. 각 구간은 개별(pointwise) 95% 구간이며 다중 비교 전체를 동시에 보장하는 구간이 아니다. 기본 budget의 두 R0 비교가 사전 지정된 주 비교이고 다른 budget·source 절편은 보조 해석이다.

두 주 비교에서 품질 차이의 구간은 0보다 크지만 비용 차이도 양수다. 따라서 R0 대비 더 높은 품질의 근거로 해석하되, 동일 비용 우위 또는 모든 목적에서의 우위로 해석하지 않는다.

## 4. 고정 baseline과 matched-random

| baseline | source-macro 평균 품질 | source-macro 상대 실제 비용 | 해석 |
|---|---:|---:|---|
| always_low | 0.605793 | 0.000000 | 모든 질문을 low로 |
| always_medium | 0.735272 | 0.069952 | 모든 질문을 medium으로 |
| always_high | 0.812702 | 1.000000 | 모든 질문을 high로 |
| train_best_single | 0.812702 | 1.000000 | train에서만 선정한 단일 모델; 두 source 모두 high |
| quality_oracle | 0.867109 | 0.142209 | 질문별 사후 outcome을 아는 비배포 상한 |

- R0는 두 source 모두 always-medium보다 품질이 낮고 비용이 높았다. 이번 고정 정책의 제한을 그대로 보여 주는 결과다.
- R1은 always-high/train-best-single보다 평균 품질이 낮지만 상대 비용도 낮다. 이를 품질 손실 없는 개선으로 보고하지 않는다.
- quality oracle은 사후 정답과 실제 비용을 사용해 최고 품질, 동률이면 낮은 비용·고정 model ID로 선택한다. 품질 상한을 이해하기 위한 비교이며 운영 가능한 라우터가 아니다.

Matched-random은 **각 방법의 validation 동결 사용 비율**을 source별 test 행 수에 맞춰 배분한 뒤 질문에 무작위로 배정한 100회(seed 0–99)다. test에서 실제 나온 라우터 사용 비율이나 실제 비용을 똑같이 맞춘 실험은 아니다. 아래 p2.5–p97.5는 무작위 배정의 경험 분포이며 bootstrap CI나 학습 seed CI가 아니다.

| 기준 사용 비율 | random 평균 품질 [p2.5, p97.5] | 실제 정책 품질 | random 평균 상대 비용 | 실제 정책 상대 비용 |
|---|---|---:|---:|---:|
| R0 CE | 0.664695 [0.660282, 0.668841] | 0.656886 | 0.136606 | 0.112373 |
| R1 success BCE | 0.761760 [0.756167, 0.766094] | 0.789141 | 0.444550 | 0.513201 |
| R2 pairwise | 0.739613 [0.734822, 0.743711] | 0.762016 | 0.274857 | 0.340807 |

R1의 실제 품질은 해당 matched-random 평균보다 높지만 실제 상대 비용도 높다. 이 차이를 동일 비용에서의 순수한 질문별 선택 효과로 단정하지 않는다.

## 5. 추가 budget은 재선정 없이 기술

| 방법 | validation budget | test source-macro 품질 | test source-macro 상대 비용 | 상한을 초과한 source |
|---|---:|---:|---:|---|
| R0 CE | 0.00 | 0.605793 | 0.000016 | xroutebench |
| R0 CE | 0.10 | 0.643613 | 0.033936 | 없음 |
| R0 CE | 0.25 | 0.656886 | 0.112373 | 없음 |
| R0 CE | 0.50 | 0.656886 | 0.112373 | 없음 |
| R0 CE | 0.75 | 0.656886 | 0.112373 | 없음 |
| R0 CE | 1.00 | 0.656886 | 0.112373 | 없음 |
| R1 success BCE | 0.00 | 0.605793 | 0.000000 | 없음 |
| R1 success BCE | 0.10 | 0.726356 | 0.062788 | 없음 |
| R1 success BCE | 0.25 | 0.754731 | 0.217173 | xroutebench |
| R1 success BCE | 0.50 | 0.789141 | 0.513201 | xroutebench |
| R1 success BCE | 0.75 | 0.806387 | 0.702664 | routerbench |
| R1 success BCE | 1.00 | 0.808331 | 0.718836 | 없음 |
| R2 pairwise | 0.00 | 0.605659 | 0.000286 | xroutebench |
| R2 pairwise | 0.10 | 0.720058 | 0.045678 | 없음 |
| R2 pairwise | 0.25 | 0.748493 | 0.201376 | xroutebench |
| R2 pairwise | 0.50 | 0.762016 | 0.340807 | xroutebench |
| R2 pairwise | 0.75 | 0.802096 | 0.643770 | 없음 |
| R2 pairwise | 1.00 | 0.801772 | 0.658091 | 없음 |

이 표는 test 결과를 보고 유리한 budget을 고르기 위한 표가 아니다. 기본 budget 0.50과 기존 R1 선정은 유지한다.

## 6. 적용 한계와 보관 증거

- 실제 공개 후보는 RouterBench의 Mistral-7B/Yi-34B/GPT-4-1106-preview와 xRouteBench의 GPT-OSS-20B/GPT-OSS-120B/Cogito-671B다. 현재 Claude 모델의 성공 확률·응답 품질·가격은 평가하지 않았다. 정확한 ID는 `evidence/protocol.json`의 arms에 보관돼 있다.
- 이 실행에는 D_gate·D_stream·D_retention_test·RouterArena 평가가 없다. 운영 재학습 gate 지원이나 실제 사용자 만족도 개선의 증거로 대체하지 않는다.
- 이 보고서 작업에서는 추가 학습·추론·λ 재선택을 수행하거나 기존 D_dev 보고서와 run 기록을 변경하지 않았다. 이 디렉터리는 기존 D_dev 선정에 대한 별도의 후속 test 증거다. 상위 결과 기록의 보관 상태 갱신과는 검증 범위를 구분한다.

결과 archive는 `gs://llm-router-mlserver/training-runs/objectives-continual-v1-A-20260924/sealed-tests-v1/sealed-evaluation.tar`다. 크기 2,293,760 bytes, SHA-256 `ad4b52f7ae3dcb6931475b5e780c71f30d16d697a1af4f7f0b184477f8eaaf8f`를 로컬에서 재검증했다. 이 archive에는 실제 model weights가 없으며 score와 평가 산출물이 들어 있다.

상위 작업은 GCS input/protocol/inference/results 4개 객체의 크기·CRC32C·generation을 독립 확인했다. 원본 transfer report의 `personal_drive.status=verified_after_flush_and_remount`와 4개 파일의 SHA·크기를 확인했고 GCS receipt와 1:1 일치함을 대조했다. 개인 Drive 경로는 `/content/drive/MyDrive/LLM_Router/routing-objectives-continual-v1-A-20260924/sealed-tests`다. 이 로컬 보고서 작업에서 Drive를 다시 원격 조회한 것은 아니다. `production_activated=false`이며 운영 게시·활성화는 수행하지 않았다.

- `summary.json`: 원본 summary를 바이트 그대로 보존. 전체 budget·source별 결과와 CI 포함.
- `verification.json`: 로컬 크기/SHA, 행 수, 동결 정책·평균 재계산 및 검증 범위.
- `evidence/`: 원본 protocol·SHA, inference/evaluation manifest·SUCCESS, transfer report.

원본 transfer report SHA-256: `003e49c75f7c55853ea4c51531a5fc0e1b36bbf9ab67bfacaacfcd874615e9eb`
