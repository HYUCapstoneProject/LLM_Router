# ModernBERT v2 Balanced Training Report

작성일: 2026-08-15
대상 run: `modernbert-router-balanced-v2`
저장 위치: `MyDrive/LLM_Router/training/modernbert-router-balanced-v2/`
학습 브랜치: `feat/data-labeling-training`

> 이 문서는 완료된 3 epoch 학습의 관측 결과와 진단만 기록한다. 라벨 규칙 자체의
> 정의는 `docs/ml/routerbench-data-labeling-spec.md` §14, 학습 계약은
> `docs/ml/modernbert-training-spec.md` §5-1이 소유한다.

## 1. 요약

3 epoch 학습은 중단 없이 완주했고 artifact도 정상 봉인되었다. 그러나 모델은
RouterBench in-distribution validation에서 **majority-class baseline을 넘지 못했고**,
RouterArena transfer에서는 **균등추측 수준에 머물렀다**.

| 결론 | 근거 | 절 |
| --- | --- | --- |
| epoch 3은 과적합이며 epoch 2가 더 낫다 | train loss 계속 하락, validation loss 상승 반전 | §4 |
| 성능 상한은 모델이 아니라 라벨에 있다 | TF-IDF baseline이 ModernBERT와 macro F1 0.03 차이 | §6 |
| 라벨이 단일 모델 우연에 좌우된다 | tier당 유효 모델 4개, `>=0.67`은 실질 "4개 중 3개" | §7 |
| 도메인 내부 easy 부족이 easy 탐지를 무너뜨린다 | 도메인 내 easy 비중 vs easy recall 상관 r=0.974 | §9 |
| **모델은 난이도가 아니라 출처 문체를 학습했다** | **출처 내부 상관 r=0.114, 예측 분산의 32.1%를 출처가 설명** | **§10** |
| RouterArena로는 튜닝 방향을 잡을 수 없다 | macro F1 0.321 ≈ 균등추측 0.326, hard 예측군에 hard 신호 없음 | §5 |

## 2. 학습 설정

`final-model/training_config.json`과 `metrics.json`에 기록된 실제 값이다.

| 항목 | 값 |
| --- | --- |
| base model | `answerdotai/ModernBERT-base` (revision `8949b909…f0c8`) |
| training version | `modernbert-router-t4-balanced-v2` |
| precision / device | FP16 / CUDA T4 |
| max length / truncation | 8,192 / 사용 (실제 잘린 행 0) |
| train batch / grad accumulation | 1 / 8 (effective batch 8) |
| sampling strategy | `balanced_effective_batch` (클래스 3/3/2 순환) |
| learning rate / warmup / weight decay | 2e-05 / 703 step (10%) / 0.01 |
| gradient checkpointing | true |
| epochs / optimizer steps | 3 / 7,038 (planned 7,038 = 완주) |
| seed | 42 |
| 학습 시간 | 20,884초 (약 5시간 48분) |
| training fingerprint | `bd9ddabb…f8181` |
| training data SHA-256 | `15a485c0…ba1e` |

## 3. 데이터

라벨 규칙은 `minimum-sufficient-tier-v2`, 전처리는
`routerbench-training-preprocess-v2-balanced`다.

| split | easy | medium | hard | 합계 |
| --- | ---: | ---: | ---: | ---: |
| train (균형화) | 6,256 | 6,256 | 6,256 | 18,768 |
| validation (자연분포) | 1,490 | 784 | 944 | 3,218 |
| test (자연분포) | 1,490 | 784 | 944 | 3,218 |
| **학습 artifact 합계** | | | | **25,204** |

validation/test는 의도적으로 자연분포를 유지하므로, validation의 majority-class
baseline accuracy는 `1,490 / 3,218 = 46.3%`다. 이 값이 이후 모든 해석의 기준선이다.

## 4. RouterBench validation 학습 곡선

epoch당 33%/66%/100% 지점에서 관측한 9개 점이다. 이 값은 관찰 전용이며 loss,
gradient, optimizer, early stopping, checkpoint 선택 어디에도 반영되지 않는다.

| obs | epoch | step | train loss | val loss | val acc | macro F1 | easy F1 | medium F1 | hard F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 782 | 1.0862 | 1.0890 | 0.4279 | 0.3706 | 0.484 | 0.149 | 0.479 |
| 2 | 1 | 1,564 | 1.0641 | **1.0487** | 0.4652 | 0.4129 | 0.546 | 0.199 | 0.494 |
| 3 | 1 | 2,346 | 1.0533 | 1.0721 | 0.4419 | 0.4254 | 0.479 | 0.311 | 0.487 |
| 4 | 2 | 3,128 | 1.0149 | 1.0637 | **0.4686** | 0.4445 | 0.549 | 0.303 | 0.481 |
| 5 | 2 | 3,910 | 1.0249 | 1.0471 | 0.4658 | 0.4452 | 0.545 | 0.309 | 0.482 |
| 6 | 2 | 4,692 | 1.0222 | 1.0601 | **0.4686** | **0.4505** | 0.544 | 0.325 | 0.482 |
| 7 | 3 | 5,474 | 0.9429 | 1.1033 | 0.4577 | 0.4341 | 0.524 | 0.300 | 0.478 |
| 8 | 3 | 6,256 | 0.9213 | 1.0920 | 0.4528 | 0.4362 | 0.517 | 0.323 | 0.468 |
| 9 | 3 | 7,038 | 0.8991 | 1.1116 | 0.4515 | 0.4352 | 0.520 | 0.316 | 0.469 |

관찰:

1. **epoch 3은 과적합 구간이다.** train loss는 1.0222 → 0.8991로 계속 내려가지만
   validation loss는 1.0601 → 1.1116으로 반전 상승한다.
2. **최종 모델은 epoch 2 시점보다 나쁘다.** macro F1 0.4505 → 0.4352 (−0.0153),
   accuracy 0.4686 → 0.4515 (−0.0171).
3. **최종 validation loss 1.1116은 `ln(3) = 1.0986`보다 크다.** 확률 출력만 놓고
   보면 균등추측보다 나쁜 상태로 끝났다.
4. **accuracy가 majority baseline(46.3%)을 넘지 못한다.** 최고점 46.86%가 baseline과
   사실상 동률이고 최종은 45.15%로 그 아래다.
5. **medium이 구조적 최약체다.** 전 구간에서 F1 0.15~0.33으로 easy(0.48~0.55),
   hard(0.47~0.49)의 절반 수준이다.

현행 학습 계약에는 best-checkpoint 선택이 없어(`affects_checkpoint_selection: false`)
무조건 마지막 step이 최종 모델이 된다. 위 1~2번은 이 계약의 직접적 결과다.

## 5. RouterArena transfer 결과

RouterArena full 8,375행에 대한 inference-only 관측이다. 학습에 어떤 방식으로도
되먹임되지 않는다.

| epoch | step | accuracy | macro F1 | under-routing | over-routing | 평균 tier 오차 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,346 | 0.3297 | 0.3014 | 0.1706 | 0.4997 | 0.9506 |
| 2 | 4,692 | **0.3582** | **0.3293** | 0.1781 | 0.4636 | 0.9100 |
| 3 | 7,038 | 0.3528 | 0.3212 | 0.1999 | 0.4473 | 0.9181 |

클래스별 F1과 예측 분포:

| epoch | easy F1 | medium F1 | hard F1 | 예측 easy | 예측 medium | 예측 hard |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4603 | 0.1272 | 0.3168 | 2,481 | 1,304 | 4,590 |
| 2 | 0.5045 | 0.1689 | 0.3146 | 2,763 | 1,326 | 4,286 |
| 3 | 0.5049 | 0.1644 | 0.2944 | 3,008 | 1,332 | 4,035 |

**RouterArena에서도 epoch 2가 최고점이다.** RouterBench validation의 결론과 독립적으로
일치하므로, 과적합 판정은 두 데이터셋에서 교차 확인되었다.

### 5-1. 예측 그룹 내부 라벨 분포 (epoch 3)

RouterArena 전체 base rate는 easy 47.4%, medium 29.1%, hard 23.5%다. 예측 그룹별
분포를 base rate와의 차이(percentage point)로 환산하면 다음과 같다.

| 예측 그룹 | n | easy | medium | hard | easy lift | medium lift | hard lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| predicted easy | 3,008 | 1,762 (58.6%) | 593 (19.7%) | 653 (21.7%) | **+11.2pp** | −9.4pp | −1.7pp |
| predicted medium | 1,332 | 594 (44.6%) | 310 (23.3%) | 428 (32.1%) | −2.8pp | −5.8pp | +8.7pp |
| predicted hard | 4,035 | 1,616 (40.0%) | 1,536 (38.1%) | 883 (21.9%) | −7.4pp | +8.9pp | **−1.6pp** |

**해석이 중요하다.** 유의미한 신호는 `predicted easy → easy (+11.2pp)` 하나뿐이다.
`predicted hard` 그룹의 hard 비율은 21.9%로 **base rate 23.5%보다 오히려 낮다**. 즉
hard 예측군은 hard에 대해 아무 정보가 없고, 오히려 RouterArena medium이 몰려 있다
(+8.9pp). medium/hard 축은 정렬되지 않은 정도가 아니라 뒤섞여 있다.

macro F1 0.3212는 이 데이터 분포에서 균등추측 기대값 0.326과 사실상 같다.
**RouterArena 기준으로는 학습된 신호가 전이되지 않았다.**

## 6. 진단 A — 성능 상한은 모델이 아니라 라벨에 있다

동일 split, 동일 텍스트로 **TF-IDF + 로지스틱 회귀** baseline을 학습해 비교했다.
단어 빈도만 사용하며 CPU에서 수 초 만에 끝나는 모델이다.

| 모델 | accuracy | macro F1 |
| --- | ---: | ---: |
| TF-IDF + LogisticRegression | 0.4344 | 0.4198 |
| ModernBERT epoch 3 (최종) | 0.4515 | 0.4352 |
| ModernBERT epoch 2 (최고점) | 0.4686 | 0.4505 |

149M 파라미터 트랜스포머를 T4로 약 6시간 학습해 얻은 이득이 bag-of-words 대비
macro F1 +0.03에 불과하다. 클래스별 이진 분리 가능성(AUC)도 낮다.

| 이진 과제 | AUC (0.5 = 신호 없음) |
| --- | ---: |
| easy vs rest | 0.6319 |
| hard vs rest | 0.6150 |
| medium vs rest | 0.5886 |

**질문 텍스트 안에 애초에 그만큼의 신호밖에 없다.** 학습률, epoch 수, 모델 크기
조정으로 넘을 수 있는 벽이 아니다.

## 7. 진단 B — 라벨 양자화 노이즈

tier별 유효 모델 수를 확인한 결과다.

| tier | 유효 모델 수 | 가능한 성공률 값 |
| --- | ---: | --- |
| low | 4 | 0 / 0.25 / 0.5 / 0.75 / 1.0 |
| medium | 4 | 0 / 0.25 / 0.5 / 0.75 / 1.0 |
| high | 3 | 0 / 0.333 / 0.667 / 1.0 |

따라서 `low_success_rate >= 0.67`은 실질적으로 **"4개 중 3개 성공"**과 같고,
`high_success_rate >= 1/3`은 **"3개 중 1개 성공"**과 같다. 모델 하나가 우연히
맞추거나 틀리면 라벨 전체가 뒤집힌다. 이는 threshold를 옮겨도 사라지지 않는
구조적 노이즈다.

threshold만 바꿔본 분포 변화:

| 기준 | easy | medium | hard | unresolved |
| --- | ---: | ---: | ---: | ---: |
| 현재 (`>=0.67` = 3/4) | 14,906 | 7,835 | 9,430 | 2,607 |
| 완화 (`>=0.5` = 2/4) | 22,395 | 6,580 | 3,919 | 1,884 |
| 강화 (`=1.0` = 4/4) | 6,869 | 5,681 | 19,261 | 2,967 |

medium은 어떤 기준에서도 5.7k~7.8k 범위에 눌린 **잔여 구간(residual band)**이다.
두 threshold 사이에 끼어 정의되므로 §4의 medium F1 최약체 현상은 필연적이다.

## 8. 라벨 정의 후보 비교 (probe)

3-class 구조는 유지한 채 **라벨을 정의하는 조건만** 바꿨을 때 학습 가능성이
달라지는지 측정했다. 라벨과 무관하게 `duplicate_group_id` 해시로 고정한 80/10/10
split을 모든 후보에 동일 적용하고, TF-IDF + LogisticRegression을 고정 용량 probe로
사용했다. 클래스 분포가 후보마다 다르므로 **chance-corrected한 Cohen's kappa**로
비교한다.

| 라벨 정의 | 유효 행 | easy | medium | hard | accuracy | macro F1 | **kappa** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A. 현재 (low≥.67, med≥.67, high≥1/3) | 32,160 | 14,906 | 7,824 | 9,430 | 0.460 | 0.4337 | 0.1598 |
| B. tier 완화 (≥.5) | 32,883 | 22,395 | 6,569 | 3,919 | 0.531 | 0.3948 | 0.1174 |
| C. tier 강화 (=1.0) | 31,800 | 6,869 | 5,670 | 19,261 | 0.474 | 0.4138 | 0.1384 |
| D. tier + 경계행 제외 | 15,405 | 6,869 | 2,682 | 5,854 | 0.489 | 0.4568 | 0.1969 |
| E. solve-rate 구간 .73/.36 | 34,764 | 12,644 | 16,151 | 5,969 | 0.485 | 0.4638 | 0.2022 |
| F. solve-rate 구간 .82/.45 | 34,764 | 8,206 | 18,206 | 8,352 | 0.468 | 0.4605 | 0.1862 |
| G. solve-rate 구간 .64/.27 | 34,764 | 16,728 | 14,128 | 3,908 | 0.520 | 0.4530 | 0.2095 |

관찰:

1. **threshold만 옮기는 것(B, C)은 현재(A)보다 나쁘다.** kappa가 0.1598 → 0.1174 /
   0.1384로 하락한다. §7의 예측과 일치한다.
2. **경계행을 제외하면(D) 개선된다.** kappa 0.1598 → 0.1969 (+23%). 단 유효 행이
   32,160 → 15,405으로 절반 이하가 되는 대가가 있다.
3. **solve-rate 기반 정의(E, F, G)가 가장 높다.** kappa 최대 0.2095 (G), 그러면서도
   **행 손실이 없다**(34,764행 전량 사용, unresolved 개념 자체가 사라짐).
   `overall_solve_rate`는 11개 모델 전체를 집계하므로 값이 12단계로 촘촘해
   §7의 양자화 노이즈가 크게 완화된다.

단 이 probe는 TF-IDF 기준의 **상대 비교**이며 ModernBERT의 절대 성능을 예측하지
않는다. 또한 solve-rate 정의는 "최소 충분 tier"가 아니라 "질문의 내재적 난이도"라는
다른 의미를 가지므로, 라우팅 목표와의 정합성은 별도 판단이 필요하다.

## 9. 진단 C — 도메인 내부 불균형이 easy 탐지를 무너뜨린다

RouterArena에서 공식 라벨이 easy인 3,972행 중 **2,210행(55.6%)이 medium/hard로
오분류**되었다(medium 594, hard 1,616). 행 단위 예측
(`epoch-03/predictions.parquet`)으로 이 2,210행의 출처를 추적했다.

### 9-1. 오분류는 카테고리에 강하게 몰려 있다

domain × 정오 교차표의 카이제곱 검정 결과 `chi2 = 301.6, dof = 8, p = 1.8e-60`으로,
카테고리별 편차는 우연이 아니다.

| domain | easy 행 | easy recall | → medium | → hard |
| --- | ---: | ---: | ---: | ---: |
| 6 Technology | 821 | **0.245** | 96 | 524 |
| 1 Philosophy and psychology | 396 | **0.295** | 36 | 243 |
| 5 Science | 774 | 0.453 | 139 | 284 |
| 7 Arts & recreation | 298 | 0.463 | 91 | 69 |
| 3 Social Science | 296 | 0.497 | 46 | 103 |
| 0 Computer science 외 | 692 | 0.529 | 64 | 262 |
| 4 Language | 209 | 0.565 | 20 | 71 |
| 8 Literature | 113 | 0.655 | 18 | 21 |
| 9 History | 373 | **0.670** | 84 | 39 |

source dataset 단위(n≥40)로 보면 격차가 더 뚜렷하다.

| 최악 | easy recall | | 최선 | easy recall |
| --- | ---: | --- | --- | ---: |
| MathQA | **0.014** | | SocialiQA | 0.787 |
| MMLU_formal_logic | 0.019 | | OpenTDB Science: Computers | 0.775 |
| LiveCodeBench | 0.089 | | OpenTDB Books | 0.768 |
| MMLUPro_math | 0.101 | | MMLUPro_history | 0.761 |
| Ethics_virtue | 0.127 | | MMLUPro_biology | 0.761 |
| MMLUPro_engineering | 0.139 | | MMLU_management | 0.627 |
| Ethics_commonsense | 0.147 | | | |
| PubMedQA | 0.161 | | | |
| MedMCQA | 0.228 | | | |

수학·논리·코드·의학·윤리 계열이 무너지고, 역사·생물·상식·문학 계열은 살아 있다.

### 9-2. 원인은 RouterBench train의 도메인 내부 라벨 분포다

v2 균형 train의 domain × label 실제 구성이다. **전역은 정확히 균형이지만 도메인
내부는 심하게 치우쳐 있다.**

| domain | easy | medium | hard | 도메인 내 easy 비중 |
| --- | ---: | ---: | ---: | ---: |
| math | 888 | 1,990 | 1,388 | **20.8%** |
| knowledge | 2,452 | 2,071 | 2,771 | 33.6% |
| commonsense | 2,767 | 2,125 | 1,999 | 40.2% |
| code | 122 | 60 | 83 | 46.0% |
| **합계** | **6,256** | **6,256** | **6,256** | 33.3% |

eval 단위로 보면 편향이 더 극단적이다. `mmlu-high-school-mathematics` 74.3% hard,
`mmlu-college-mathematics` 69.7% hard, `mmlu-moral-scenarios` 68.4% hard,
`mmlu-professional-medicine` 60.1% hard, `mmlu-formal-logic` 56.0% hard.
반면 `arc-challenge` 51.3% easy, `mbpp` 46.0% easy다.

**이 편향은 RouterArena만의 문제가 아니다.** RouterBench 자체 validation에서
RouterBench 자체 라벨로 재봐도 같은 붕괴가 재현된다(TF-IDF probe 기준).

| domain | train 내 easy 비중 | validation easy recall |
| --- | ---: | ---: |
| math | 0.208 | **0.084** |
| knowledge | 0.336 | 0.532 |
| commonsense | 0.402 | 0.550 |
| code | 0.460 | 0.812 |

n≥200 도메인 4개에서 **상관 r = 0.974 (p = 0.026)**. 즉 어떤 도메인의 easy를
탐지하는 능력은 그 도메인에 easy 학습 데이터가 얼마나 있었는지로 거의 결정된다.

### 9-3. 모델이 실제로 학습한 것은 표층 휴리스틱이다

가장 확신을 갖고 hard로 틀린 easy 질문들이다.

| 출처 | `prob_easy` | 질문 요지 |
| --- | ---: | --- |
| MMLUPro_math | 0.0013 | 수열 5, 9, 13, 17, 21…의 규칙 고르기 (초등 수준 등차수열) |
| MMLUPro_business | 0.0019 | `(7/10) EOM` 이라는 거래 용어의 뜻 설명 |
| MMLUPro_business | 0.0012 | 효용함수 기대값 계산 |
| MedMCQA | 0.0019 | 효소 반응 농도 계산 |

첫 두 사례가 특히 진단적이다. 등차수열 규칙 고르기는 명백히 쉬운 문제인데
`prob_hard = 0.993`이고, `(7/10) EOM`은 수학 문제가 아니라 용어 질문인데 분수 표기
때문에 hard로 밀렸다. **모델은 난이도가 아니라 "숫자·수식 표기가 있으면 어렵다"는
표층 신호를 학습했다.** math 도메인의 79.2%가 non-easy였으므로 이는 학습 데이터에
대해 합리적인 귀결이다.

### 9-4. 새 데이터 수집 없이 고칠 수 있는 부분이 있다

전역 균형화(각 클래스 6,256행)가 **easy 행 5,670개를 버렸고**, 그중 상당수가 정확히
부족한 도메인의 easy다.

| domain | 균형 train 사용 easy | 균형화로 버려진 easy | 전체 가용 easy |
| --- | ---: | ---: | ---: |
| math | 888 | **825** | 1,713 |
| knowledge | 2,452 | 2,277 | 4,729 |
| commonsense | 2,767 | 2,568 | 5,335 |

math의 버려진 easy 825행을 되살리면 math 도메인 구성은 easy 1,713 / medium 1,990 /
hard 1,678이 되어 **도메인 내부에서도 거의 균형**을 이룬다. 즉 math에 관해서는
데이터를 새로 모을 필요가 없고, **균형화 기준을 전역에서 domain × label 층화로
바꾸기만 하면 된다.**

단 이는 math·knowledge·commonsense에 해당하는 이야기다. RouterArena에서 무너진
의학(PubMedQA, MedMCQA), 윤리(Ethics_*), 형식논리(MMLU_formal_logic) 계열은
RouterBench에 애초에 표본이 적으므로, 이쪽은 실제 데이터 보강이 필요하다.

## 10. 진단 D — 모델은 난이도가 아니라 출처 문체를 학습했다

§9는 easy 오분류만 다뤘다. 9개 칸 전체와 오분류 *방향*까지 분석하면 더 근본적인
결론이 나온다.

### 10-1. 전체 혼동행렬

| 실제 \ 예측 | easy | medium | hard | recall |
| --- | ---: | ---: | ---: | ---: |
| easy (3,972) | 1,762 | 594 | 1,616 | 44.4% |
| medium (2,439) | 593 | 310 | 1,536 | **12.7%** |
| hard (1,964) | 653 | 428 | 883 | 45.0% |

medium recall이 12.7%로 사실상 붕괴 상태이고, medium의 63.0%가 hard로 밀린다.

### 10-2. 편향은 출처별로 방향이 고정되어 있다

`평균 예측 tier − 평균 실제 tier`(easy=0, medium=1, hard=2)로 출처별 체계적 편향을
계산했다(n≥40).

| hard 쪽으로 밀림 | 편향 | | easy 쪽으로 밀림 | 편향 |
| --- | ---: | --- | --- | ---: |
| Ethics_virtue | **+1.60** | | ChessInstruct | **−1.47** |
| Ethics_commonsense | +1.47 | | WMT19-de-en | −1.37 |
| MMLU_formal_logic | +1.42 | | WMT19-gu-en | −1.33 |
| MathQA | +1.40 | | QANTA_Philosophy | −1.00 |
| PubMedQA | +1.33 | | QANTA_Social Science | −0.98 |
| MMLUPro_math | +1.06 | | QANTA_Literature | −0.94 |
| Ethics_deontology | +1.02 | | SuperGLUE-ClozeTest | −0.90 |
| MMLUPro_engineering | +0.95 | | QANTA_History | −0.78 |

easy 쪽으로 밀린 출처가 특히 진단적이다. WMT19(기계번역)는 실제 tier가 2.00(전부
hard)인데 예측은 0.63~0.67이다. 번역 원문은 평범한 산문이라 "어려워 보이는" 표층
신호가 없다. ChessInstruct도 마찬가지로 실제 1.85 → 예측 0.38이다.

**즉 §9-3에서 관찰한 "숫자·수식이 있으면 hard"의 반대 방향도 성립한다: 평범한
산문이면 easy.** 모델은 난이도가 아니라 표기 형식을 읽는다.

### 10-3. 오분류 방향은 실제 라벨이 아니라 출처가 결정한다

같은 실제 라벨 안에서 두 오분류 방향을 갈라 보면, 갈리는 기준이 출처다.

**실제=medium** (오분류 2,129건 중 →easy가 27.9%)

| →hard로 쏠린 출처 | →easy 비율 | | →easy로 쏠린 출처 | →easy 비율 |
| --- | ---: | --- | --- | ---: |
| MMLU_formal_logic (28) | 0.0% | | ChessInstruct_mcq (44) | 97.7% |
| MathQA (81) | 1.2% | | SuperGLUE-RC (54) | 88.9% |
| LiveCodeBench (175) | 1.7% | | MMLUPro_history (92) | 84.8% |
| MMLUPro_math (108) | 1.9% | | SuperGLUE-QA (41) | 82.9% |

**실제=hard** (오분류 1,081건 중 →easy가 60.4%)

| →medium으로 쏠린 출처 | →easy 비율 | | →easy로 쏠린 출처 | →easy 비율 |
| --- | ---: | --- | --- | ---: |
| QANTA_Fine Arts (44) | 18.2% | | SuperGLUE-ClozeTest (27) | 100.0% |
| QANTA_History (64) | 34.4% | | ArcMMLU (31) | 96.8% |
| QANTA_Literature (142) | 40.1% | | MMLUPro_history (58) | 96.6% |

**실제=easy** (오분류 2,210건 중 →medium이 26.9%)

| →hard로 쏠린 출처 | →medium 비율 | | →medium으로 쏠린 출처 | →medium 비율 |
| --- | ---: | --- | --- | ---: |
| MathQA (72) | 0.0% | | AsDiv (26) | 92.3% |
| LiveCodeBench (41) | 0.0% | | GeoBench (100) | 71.0% |
| Ethics_commonsense (64) | 3.1% | | MMLUPro_history (37) | 62.2% |

**같은 출처가 실제 라벨과 무관하게 항상 같은 방향으로 간다.** MathQA·LiveCodeBench·
formal_logic은 실제가 easy든 medium이든 hard로 가고, MMLUPro_history·ChessInstruct·
SuperGLUE-RC는 실제가 easy든 medium이든 hard든 easy로 간다.

### 10-4. 결정적 검정 — 출처 내부에서는 난이도를 전혀 구분하지 못한다

| 지표 | 값 |
| --- | ---: |
| 전체 상관 (실제 tier vs 예측 tier) | r = **0.101** |
| 출처 내부(within-dataset) 상관 | r = **0.114** |
| 예측 tier 분산 중 출처로 설명되는 비율 | **32.1%** |

같은 데이터셋 안에서 실제 라벨별 평균 예측 tier를 보면 단조성이 거의 없다.
모델이 난이도를 읽는다면 easy < medium < hard로 증가해야 한다.

| 출처 | 실제 easy | 실제 medium | 실제 hard | 단조? |
| --- | ---: | ---: | ---: | --- |
| PubMedQA | 1.63 | 1.85 | 1.63 | ✗ (hard = easy) |
| MMLUPro_math | 1.71 | 1.95 | 1.81 | ✗ |
| MMLUPro_history | 0.33 | 0.38 | 0.27 | ✗ (hard < easy) |
| NarrativeQA | 1.27 | 0.79 | 1.26 | ✗ |
| MedMCQA | 1.26 | 1.55 | 1.48 | ✗ |
| MusicTheoryBench | 0.95 | 1.62 | 1.52 | ✗ |
| LiveCodeBench | 1.82 | 1.97 | 1.98 | ○ (약함) |
| GeoBench | 0.50 | 0.60 | 0.77 | ○ (약함) |

**모델은 출처마다 거의 상수를 출력하며, 실제 난이도는 그 상수를 거의 움직이지
못한다.** 모델은 `source_dataset`을 본 적이 없으므로, 문체·형식·주제어에서 출처를
사실상 식별한 뒤 그 출처의 학습 시 사전확률을 출력하는 것이다.

### 10-5. 맞힌 것도 대부분 우연이다

출처의 최빈 실제 라벨과 모델의 최빈 예측이 일치하는지로 나눠보면:

| 일치 여부 | 출처 수 | 행 수 | 평균 정확도 |
| --- | ---: | ---: | ---: |
| 일치 | 21 | 2,929 | **0.530** |
| 불일치 | 35 | 4,932 | **0.244** |

정확도가 높은 출처는 예외 없이 "그 출처가 원래 한 라벨로 쏠려 있고, 모델의 상수가
우연히 그 라벨과 같은" 경우다.

| 출처 | 정확도 | 모델 최빈예측 | 그 출처의 최빈 실제라벨 비중 |
| --- | ---: | --- | ---: |
| GeoGraphyData_100k | 0.800 | easy | 0.800 |
| OpenTDB Books | 0.708 | easy | 0.854 |
| OpenTDB Science: Computers | 0.696 | easy | 0.887 |
| MMLUPro_biology | 0.671 | easy | 0.818 |
| OpenTDB History | 0.579 | easy | **0.965** |

마지막 행이 특히 시사적이다. OpenTDB History는 96.5%가 easy인데 모델 정확도는
57.9%다. **무조건 easy라고 답하는 것보다도 나쁘다.**

정답 칸의 출처 구성도 같은 이야기를 한다.

- easy 정답 1,762건: ArcMMLU 161, GeoBench 158, MMLUPro_history 118, MMLUPro_biology 89
- hard 정답 883건: NarrativeQA 211, LiveCodeBench 163, MMLUPro_engineering 37, PubMedQA 37
- medium 정답 310건: 최다가 MusicTheoryBench 35로, 어떤 출처도 medium으로 안정적이지 않다

**medium이 무너지는 이유가 여기서 설명된다.** 모델의 예측이 출처 상수라면, medium을
맞히려면 "대부분이 medium인 출처"가 있어야 하는데 그런 출처가 존재하지 않는다.
medium은 어느 주제에나 흩어져 있는 난이도이므로 출처 문체로는 잡히지 않는다.

### 10-6. §9와의 관계

§9(도메인 내부 라벨 불균형)와 모순되지 않고 그 메커니즘을 설명한다. 학습 데이터에서
주제와 난이도가 상관되어 있었기 때문에(math의 79.2%가 non-easy) 모델은 **주제를
난이도의 대리 변수로 학습**했다. 따라서 §9-4의 `domain × label` 층화 균형화는 바로
이 허위 상관을 끊는 조치이며, 본 절의 발견은 그 처방의 근거를 강화한다.

다만 층화만으로 충분한지는 불확실하다. 모델이 주제 신호를 쓰지 못하게 되면 남는
것이 §6에서 측정한 AUC 0.59~0.63 수준의 약한 텍스트 신호뿐일 수 있다.

## 11. 진단 E — RouterBench 커버리지 점검: 부재와 결핍은 다른 문제다

§9~10의 처방(도메인 층화)이 어디까지 유효한지 판단하려면, 무너진 카테고리가
RouterBench에 **얼마나 있었는지** 확인해야 한다. RouterBench의 65개 `eval_name`을
주제 클러스터로 묶어 RouterArena 성적과 나란히 놓았다.

| 클러스터 | RB train 행 | RB easy 비중 | RA 행 | RA easy recall | RA 정확도 | 편향 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 번역 (WMT19) | **0** | – | 151 | – | 0.099 | −1.34 |
| 음악 (MusicTheory) | **0** | – | 221 | 0.325 | 0.348 | +0.62 |
| 장문독해 (NarrativeQA) | **0** | – | 383 | 0.364 | 0.574 | −0.61 |
| 체스 (ChessInstruct) | **0** | – | 148 | 0.939 | 0.236 | −0.95 |
| 지리 | 113 | **53.1%** | 374 | **0.660** | 0.570 | +0.22 |
| 형식논리 | 138 | 35.5% | 91 | **0.019** | 0.110 | +1.42 |
| 생물 | 248 | 40.7% | 143 | **0.761** | 0.671 | +0.33 |
| 공학 | 311 | 22.2% | 280 | 0.139 | 0.182 | +0.95 |
| 코드 | 468 | 40.4% | 385 | 0.089 | 0.434 | +0.64 |
| 역사 | 492 | 43.3% | 497 | **0.698** | 0.400 | −0.38 |
| 윤리 | 736 | 23.9% | 283 | 0.168 | 0.187 | +1.22 |
| 의학 | 903 | 32.6% | 991 | 0.208 | 0.249 | +1.05 |
| 수학 | **4,758** | 20.3% | 451 | 0.184 | 0.164 | +0.99 |

### 11-1. 양이 아니라 easy 비중이 성능을 결정한다

커버리지가 있는 9개 클러스터에서:

| 예측 변수 | easy recall과의 상관 |
| --- | --- |
| RouterBench 행 수 | Pearson **r = −0.225** (p = 0.560), Spearman +0.033 |
| RouterBench easy 비중 | Pearson **r = +0.650** (p = 0.058), Spearman +0.483 |

**절대량은 성능을 전혀 예측하지 못하고(오히려 음의 부호), easy 비중이 예측한다.**
가장 극명한 대비는 다음이다.

- **지리**: RouterBench 113행(최소급)인데 easy가 53.1% → easy recall **0.660**
- **수학**: RouterBench 4,758행(최다)인데 easy가 20.3% → easy recall **0.184**

데이터가 42배 많은 수학이 지리보다 easy를 3.6배 못 맞춘다. 따라서 **"수학·의학·윤리
데이터를 더 모으자"는 처방은 틀렸다.** 이들은 양이 아니라 구성이 문제이며, §9-4의
층화 균형화가 정확한 처방이다.

### 11-2. 다만 절대량 하한은 존재한다

형식논리는 easy 비중이 35.5%로 나쁘지 않은데도 easy recall이 0.019로 최악이다.
RouterBench 행이 138개(easy 49개)뿐이기 때문이다. 즉 **easy 비중이 충분해도 절대량이
임계 이하면 무너진다.** 코드(468행, easy 40.4%, recall 0.089)도 같은 구간에 있다.

층화 균형화로 해결되는 것과 실제 데이터 보강이 필요한 것을 구분하면 다음과 같다.

| 처방 | 대상 | 근거 |
| --- | --- | --- |
| 층화 균형화로 해결 가능 | 수학, 의학, 윤리, 공학 | 행 수는 충분(311~4,758), easy 비중만 부족 |
| 실제 데이터 보강 필요 | 형식논리, 코드 | easy 비중은 있으나 절대량이 138~468행 |
| 신규 수집 필요 (RB에 없음) | 번역, 음악, 체스, 장문독해 | RouterBench 행 **0** |

### 11-3. 커버리지가 0이면 예측 방향을 통제할 수 없다

RouterBench에 없는 4개 클러스터의 결과는 일관되지 않다. 모델이 표층 문체로 추정할
뿐이므로(§10-2), 우연히 맞기도 하고 크게 빗나가기도 한다.

- **장문독해(NarrativeQA)**: 정확도 0.574로 준수하다. 실제 라벨의 90.6%가 hard이고
  모델도 hard로 미는 방향이 우연히 일치했다.
- **번역(WMT19)**: 정확도 0.099로 최악이다. 실제는 전부 hard인데 원문이 평범한
  산문이라 모델이 easy로 민다(편향 −1.34).
- **체스**: easy recall 0.939로 높아 보이지만 정확도는 0.236이다. 모델이 거의 모든
  체스 문제를 easy로 밀어서(편향 −0.95) easy만 우연히 맞을 뿐이다.

**easy recall 하나만 보면 오독할 수 있다는 점에 유의해야 한다.** 체스처럼 한쪽으로
쏠린 예측은 특정 클래스의 recall만 부풀린다.

## 12. 균형화 전략 실험 — 전역 균형화가 math 결핍의 원인이었다

§9~11의 처방(층화 균형화)을 실제로 구현해 TF-IDF probe로 검증했다. 결과는 **처방을
수정해야 함**을 보여준다.

### 12-1. 결정적 사실: math의 easy 결핍은 균형화가 만들었다

| math 도메인 | easy | medium | hard | 최대/최소 비 |
| --- | ---: | ---: | ---: | ---: |
| 자연분포 (균형화 전) | **1,713** | 1,990 | 1,678 | **1.19** |
| 전역 균형화 후 (현행 v2) | **888** | 1,990 | 1,388 | 2.24 |

자연분포에서 math는 이미 거의 균형(비율 1.19)이다. **전역 균형화가 easy 총량을
6,256으로 맞추려고 math의 easy를 1,713 → 888로 절반 가까이 깎았다.** §9에서 관찰한
"math의 easy 비중 20.8%"는 데이터의 성질이 아니라 균형화가 만들어낸 인공물이다.

실제로 모든 도메인의 자연 최대/최소 비율은 1.19~2.70으로, 애초에 층화가 필요할 만큼
치우쳐 있지 않았다.

### 12-2. 균형화 전략 비교 (TF-IDF probe, 동일 validation)

| 전략 | train 행 | accuracy | macro F1 | kappa | medium F1 | math easy recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A. 현행 전역 균형화 | 18,768 | 0.4344 | 0.4198 | 0.1388 | **0.339** | **0.084** |
| **B. 자연분포 전량** | **25,724** | **0.4518** | **0.4205** | **0.1397** | 0.294 | **0.327** |
| C. 자연분포 + class weight | 25,724 | 0.3763 | 0.3768 | 0.1027 | **0.363** | 0.079 |
| D. domain 층화 (완전 균등) | 17,832 | 0.4121 | 0.4028 | 0.1135 | – | 0.397 |
| E. eval_name 층화 (완전 균등) | 16,770 | 0.3710 | 0.3653 | 0.0604 | – | 0.355 |

**B(균형화를 아예 하지 않음)가 accuracy·macro F1·kappa·math easy recall 4개 지표에서
최선이다.** 현행 A는 데이터의 27%(6,956행)를 버리면서 math easy recall만 1/4로
떨어뜨렸다.

C가 특히 진단적이다. class weight로 medium/hard를 올리자 medium F1은 0.363으로
가장 좋아졌지만 **math easy recall이 0.079로 다시 붕괴했다.** 즉 easy 행을 버리든
non-easy에 가중치를 주든, **easy 쪽 신호를 약화시키는 모든 조치가 math easy 실명을
유발한다.** math의 easy는 모델의 easy 판정 중 가장 한계적인 영역이기 때문이다.

D·E의 완전 층화는 math easy recall은 가장 높지만(0.355~0.397) 전체 지표가 크게
나빠진다. 주제 신호를 완전히 제거하면 §6에서 측정한 AUC 0.59~0.63의 약한 텍스트
신호만 남기 때문이며, 이는 §10-6에서 예고한 위험이 실현된 것이다.

### 12-3. 수정된 처방

§9-4와 §11의 "층화 균형화" 권고를 **"균형화 제거"로 대체한다.** 주제와 난이도의
상관은 허위 상관이 아니라 실제 신호이므로(수학 문제는 실제로 약한 모델에게 어렵다)
제거 대상이 아니다. 문제는 균형화가 그 위에 인공적인 easy 결핍을 덧씌운 것이었다.

medium F1 손실(0.339 → 0.294)은 감수한다. medium은 §7에서 확인한 잔여 구간이라
어떤 전략에서도 취약하며, C에서 보듯 medium을 살리는 조치는 다른 모든 지표를 해친다.

`select_stratified_balanced_train_groups`는 구현해 두었으나
(`balanced_train.strategy = "stratified"`), 위 결과에 따라 기본값은
`"global"`을 유지하고 향후 비교 실험용으로만 남긴다.

### 12-4. 외부 데이터 확충의 제약 — 라벨 의미가 재현 불가능하다

§11-2의 "실제 보강 필요"(형식논리, 코드)와 "신규 수집 필요"(번역·음악·체스·장문독해)
항목에 대해 확충 경로를 조사했다.

**후보 데이터셋은 존재한다.** 모두 영어이며 자체 난이도 라벨을 가진다.

| 대상 | 후보 | 규모 | 자체 난이도 라벨 |
| --- | --- | ---: | --- |
| 코드 | APPS (`codeparrot/apps`) | 10,000 | introductory / interview / competition (3단계) |
| 코드 | DeepMind CodeContests | 다수 | 대회 난이도 rating |
| 형식논리 | ReClor | 6,138 | test split만 Test-E / Test-H (2단계) |
| 형식논리 | LogiQA 2.0 | 8,678 | 없음 |
| 형식논리 | FOLIO | 1,435 | 없음 |

**그러나 이 라벨들을 그대로 병합할 수 없다.** RouterBench 라벨은 "11개 모델 중
최소 충분 tier"이고, APPS의 라벨은 "어느 대회 난이도에서 출제되었는가"다. 서로 다른
구성 개념이며, 이를 같은 클래스로 합치면 **§5-1에서 RouterArena와 관찰한 라벨 의미
불일치를 학습셋 내부에 그대로 재현**하게 된다.

의미를 맞추려면 새 질문에 동일한 11개 모델을 실행해 tier 성공률을 다시 산출해야
하는데, `model_tiers.json`의 구성 모델은 다음과 같다.

```text
low    : mistral-7b-chat, claude-instant-v1, code-llama-instruct-34b-chat, WizardLM-13B-V1.2
medium : claude-v1, llama-2-70b-chat, claude-v2, gpt-3.5-turbo-1106
high   : mixtral-8x7b-chat, Yi-34B-Chat, gpt-4-1106-preview
```

`claude-instant-v1`, `claude-v1`, `claude-v2`는 모두 서비스 종료되었고
`gpt-3.5-turbo-1106`, `gpt-4-1106-preview`도 폐기된 snapshot이다. **따라서 RouterBench
tier 정의를 새 데이터에 대해 정확히 재현하는 것은 현재 불가능하다.** 나머지 open
model 6개를 자체 호스팅해 근사할 수는 있으나, tier 경계가 달라지므로 기존 라벨과의
호환성이 깨진다.

**결론:** 외부 데이터 확충은 현시점에서 실행 가능한 선택지가 아니다. §12-3의
"균형화 제거"가 새 데이터 없이 얻을 수 있는 최선이며, 형식논리·코드의 절대량 부족은
당분간 알려진 한계로 기록해 둔다.

### 12-5. 커버리지 0 카테고리는 학습에 추가하지 않는다

번역·음악·체스·장문독해는 RouterBench에 없고 RouterArena에만 있다. 이들을 학습에
추가할 근거가 "RouterArena에 있으니까"라면, 그것은 **평가셋에 학습 분포를 맞추는
행위**이며 `routerarena-routing-evaluation-spec.md`가 금지한 것이다.

포함 여부의 올바른 기준은 **production 트래픽 분포**다. 이 서비스는 사용자 질의를
받아 라우팅하는 챗 제품이므로 기계번역 원문·체스 기보·음악 이론 문항은 대표적인
질의로 보기 어렵다. 따라서 이 4개 카테고리는 학습에 추가하지 않고, RouterArena
평가에서도 **해당 카테고리를 분리 보고**해 전체 지표를 왜곡하지 않도록 한다.

`NarrativeQA`(장문독해 383행)는 RouterArena full의 4.6%를 차지하고 정확도 0.574로
전체 평균보다 높아, 포함 시 지표를 과대평가하게 하는 점도 분리 보고의 근거다.

## 13. 권고

우선순위 순이다. 1번과 2번은 새 데이터 수집 없이 즉시 실행 가능하다.

1. **epoch 2 checkpoint로 교체하거나 best-checkpoint 선택을 도입한다.** 재학습 없이
   macro F1 +0.015를 즉시 회수한다. RouterBench와 RouterArena 양쪽에서 epoch 2가
   최고점임이 교차 확인되었다.
2. **`balanced_train.enabled`를 끄고 자연분포 전량(25,724행)으로 학습한다.**
   §12-2에서 accuracy·macro F1·kappa·math easy recall 4개 지표 모두 현행보다
   좋았다. 버려지던 6,956행을 되찾고 math easy recall이 0.084 → 0.327이 된다.
   medium F1은 0.339 → 0.294로 감수한다. `preprocessing_version`을 올려 기존 v2
   artifact는 보존한다.
3. **라벨 정의를 solve-rate 기반(E/G)으로 바꾸는 실험을 한다.** 3-class 구조를
   유지하면서 probe kappa 기준 +31% 개선이 관측되었고 데이터 손실도 없다.
   `preprocessing_version`을 올려 기존 v2 artifact를 덮어쓰지 않는다.
4. **형식논리·코드는 실제 데이터 보강을 검토한다.** §11-2에서 확인했듯 easy 비중은
   충분한데 절대량이 138~468행뿐이다. 단 보강 데이터는 RouterBench와 **동일한 라벨
   의미**(11개 모델의 최소 충분 tier)를 가져야 하므로 외부 데이터셋의 자체 난이도
   라벨을 그대로 병합해서는 안 된다(§12-4).
5. **그래도 개선이 없으면 과제 정의를 재검토한다.** §6의 AUC 0.59~0.63은 "질문
   텍스트만으로 최소 충분 tier를 예측한다"는 전제 자체의 한계를 시사한다.

### 하지 말아야 할 것

**RouterArena 지표를 보고 라벨 threshold나 hyperparameter를 조정하지 않는다.**
`docs/ml/routerarena-routing-evaluation-spec.md`가 명시한 금지 사항이며, 실무적으로도
§5-1에서 확인했듯 macro F1이 균등추측과 동일하고 hard 예측군에 hard 신호가 없어
따라갈 방향 자체가 존재하지 않는다.

§9의 카테고리 분석은 이 금지와 충돌하지 않는다. 근거가 되는 인과(도메인 내부 라벨
불균형 → easy recall 저하)를 **RouterBench 자체 validation에서 독립적으로 재현**했고
(§9-2), 처방도 RouterBench 전처리 층화이지 RouterArena 점수에 맞춘 threshold 조정이
아니기 때문이다. RouterArena는 가설을 제기하는 데 쓰였고 검증은 RouterBench에서 했다.

## 13. Artifact 무결성

| 항목 | 상태 |
| --- | --- |
| `final-model/_SUCCESS` | 존재 |
| `final-model/artifact_manifest.json` | 존재 (size + SHA-256 검증) |
| checkpoint reload smoke | `reload_smoke_passed: true` |
| optimizer steps | 7,038 / planned 7,038 (완주) |
| 중단·재개 이력 | `resumed_from_step: null` (무중단 완주) |
| RouterArena epoch 관측 | epoch-01 / 02 / 03 전부 봉인 완료 |
| RouterBench validation 관측 | observation-01 ~ 09 전부 봉인 완료 |

학습 자체의 재현성과 무결성에는 문제가 없다. 본 보고서의 지적은 전적으로 라벨
정의와 checkpoint 선택 정책에 관한 것이다.
