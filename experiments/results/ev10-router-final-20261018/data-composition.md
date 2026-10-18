# EV-10 데이터 구성 — 카테고리별 분포

00_prepare의 train·dev·test에서 v2 제외 문항, 평가 제외 문항(dev·test의 영어가 아닌 문항과 문서 제공형 SQuAD·BoolQ), 집단 풀 확률이 빈 행을 뺀 **실제로 쓰는 행**이다. test는 문항 수만 세고 결과(필요 단계)는 읽지 않았다. 표의 숫자는 `categories.py composition`이 만든다.

## 0. 필요 단계 비율 (train·dev)

필요 단계 = 집단 풀 확률이 .8 이상인 가장 싼 단계(계획서 2.2). test는 결과라 08에서 연다.

| 범위 | train low / medium / high | dev low / medium / high |
|---|---|---|
| 전체(행 그대로) | 54.7% / 26.5% / 18.8% | 57.7% / 25.4% / 16.9% |
| 전체(원천 같은 비중, 학습이 보는 비율) | 61.0% / 23.6% / 15.4% | 60.3% / 24.3% / 15.5% |
| routerbench | 43.6% / 30.7% / 25.7% | 45.5% / 29.6% / 24.9% |
| xroutebench | 76.2% / 14.2% / 9.6% | 71.8% / 18.7% / 9.5% |
| llmrouterbench | 53.3% / 28.7% / 18.0% | 53.4% / 27.9% / 18.7% |
| sprout | 70.9% / 20.7% / 8.4% | 70.4% / 20.8% / 8.8% |

## 1. 과제 유형별 (원천 합계)

| 과제 유형 | train 행 (비율) | dev 행 (비율) | test 행 (비율) | 들어 있는 원천 |
|---|---:|---:|---:|---|
| 수학 | 4,670 (13.5%) | 797 (10.5%) | 1,803 (10.2%) | routerbench 5,440, xroutebench 1,709, llmrouterbench 121 |
| 코드 | 747 (2.2%) | 164 (2.2%) | 1,173 (6.6%) | llmrouterbench 938, xroutebench 839, routerbench 307 |
| 지식 객관식 | 8,260 (23.9%) | 1,438 (18.9%) | 2,833 (16.0%) | routerbench 10,024, xroutebench 1,352, llmrouterbench 1,155 |
| 상식·추론 | 7,905 (22.9%) | 1,320 (17.4%) | 2,377 (13.5%) | routerbench 9,325, xroutebench 2,277 |
| 독해 | 472 (1.4%) | 0 (0.0%) | 0 (0.0%) | xroutebench 472 |
| 전문가 과학·난제 | 905 (2.6%) | 305 (4.0%) | 702 (4.0%) | llmrouterbench 1,571, sprout 341 |
| 사실 회상 | 2,114 (6.1%) | 659 (8.7%) | 1,552 (8.8%) | llmrouterbench 4,325 |
| 열린 대화·지시 | 9,500 (27.5%) | 2,908 (38.3%) | 7,222 (40.9%) | sprout 19,117, llmrouterbench 457, routerbench 56 |
| **합계** | **34,573** | **7,591** | **17,662** | |

학습 손실은 원천마다 같은 총 가중치를 주므로(계획서 2.4), 학습에서 실제로 받는 비중은 아래 2절의 원천 안 비율 × 1/4이다.

## 2. 원천별 카테고리

### routerbench

| 카테고리 | 과제 유형 | train | dev | test | 필요 단계 low/medium/high (train) | (dev) |
|---|---|---:|---:|---:|---|---|
| code | 코드 | 224 (1%) | 39 (1%) | 44 (1%) | 46% / 18% / 36% | 41% / 21% / 38% |
| commonsense | 상식·추론 | 6,901 (37%) | 1,147 (37%) | 1,277 (37%) | 43% / 35% / 22% | 44% / 33% / 22% |
| conversation | 열린 대화·지시 | 39 (0%) | 8 (0%) | 9 (0%) | 59% / 15% / 26% | 62% / 25% / 12% |
| knowledge | 지식 객관식 | 7,417 (40%) | 1,237 (40%) | 1,370 (40%) | 50% / 29% / 21% | 51% / 29% / 19% |
| math | 수학 | 4,025 (22%) | 670 (22%) | 745 (22%) | 32% / 27% / 41% | 36% / 24% / 39% |

### xroutebench

| 카테고리 | 과제 유형 | train | dev | test | 필요 단계 low/medium/high (train) | (dev) |
|---|---|---:|---:|---:|---|---|
| aime_2020_2024 | 수학 | 64 (2%) | 11 (3%) | 13 (0%) | 91% / 2% / 8% | 91% / 9% / 0% |
| arc_challenge | 상식·추론 | 266 (10%) | 45 (12%) | 500 (14%) | 91% / 7% / 2% | 93% / 7% / 0% |
| boolq | 독해 | 234 (9%) | 0 (0%) | 0 (0%) | 91% / 6% / 4% | – |
| commonsense_qa | 상식·추론 | 234 (9%) | 42 (11%) | 50 (1%) | 76% / 12% / 12% | 74% / 19% / 7% |
| gsm8k | 수학 | 263 (10%) | 47 (12%) | 500 (14%) | 76% / 10% / 14% | 70% / 11% / 19% |
| hellaswag | 상식·추론 | 237 (9%) | 40 (11%) | 50 (1%) | 75% / 14% / 11% | 68% / 22% / 10% |
| human_eval | 코드 | 77 (3%) | 13 (3%) | 16 (0%) | 91% / 5% / 4% | 85% / 15% / 0% |
| math | 수학 | 264 (10%) | 47 (12%) | 500 (14%) | 48% / 44% / 8% | 47% / 40% / 13% |
| mbpp | 코드 | 198 (7%) | 35 (9%) | 500 (14%) | 62% / 18% / 21% | 46% / 23% / 31% |
| mmlu | 지식 객관식 | 265 (10%) | 46 (12%) | 500 (14%) | 79% / 15% / 6% | 76% / 20% / 4% |
| mmlu_pro | 지식 객관식 | 34 (1%) | 7 (2%) | 500 (14%) | 56% / 26% / 18% | 71% / 14% / 14% |
| openbook_qa | 상식·추론 | 267 (10%) | 46 (12%) | 500 (14%) | 82% / 15% / 4% | 87% / 13% / 0% |
| squad | 독해 | 238 (9%) | 0 (0%) | 0 (0%) | 76% / 5% / 19% | – |

### llmrouterbench

| 카테고리 | 과제 유형 | train | dev | test | 필요 단계 low/medium/high (train) | (dev) |
|---|---|---:|---:|---:|---|---|
| aime | 수학 | 12 (0%) | 7 (1%) | 18 (0%) | 42% / 58% / 0% | 29% / 71% / 0% |
| arenahard | 열린 대화·지시 | 0 (0%) | 0 (0%) | 457 (12%) | – | – |
| gpqa | 전문가 과학·난제 | 98 (3%) | 31 (3%) | 69 (2%) | 50% / 44% / 6% | 42% / 48% / 10% |
| hle | 전문가 과학·난제 | 662 (18%) | 213 (19%) | 498 (13%) | 66% / 13% / 21% | 66% / 11% / 23% |
| livecodebench | 코드 | 248 (7%) | 77 (7%) | 613 (17%) | 54% / 37% / 8% | 58% / 35% / 6% |
| livemathbench | 수학 | 42 (1%) | 15 (1%) | 27 (1%) | 81% / 14% / 5% | 93% / 7% / 0% |
| mmlupro | 지식 객관식 | 544 (15%) | 148 (13%) | 463 (13%) | 80% / 16% / 4% | 87% / 9% / 3% |
| simpleqa | 사실 회상 | 2,114 (57%) | 659 (57%) | 1,552 (42%) | 42% / 35% / 23% | 41% / 36% / 23% |

### sprout

| 카테고리 | 과제 유형 | train | dev | test | 필요 단계 low/medium/high (train) | (dev) |
|---|---|---:|---:|---:|---|---|
| Idavidrein/gpqa/gpqa_extended | 전문가 과학·난제 | 145 (2%) | 61 (2%) | 135 (2%) | 27% / 31% / 42% | 30% / 39% / 31% |
| openhermes/teknium | 열린 대화·지시 | 9,461 (98%) | 2,900 (98%) | 6,756 (98%) | 72% / 21% / 8% | 71% / 20% / 8% |

## 3. RouterArena (진단용, 결과 없음)

| 분야 | 문항 |
|---|---:|
| 0 Computer science, information, and general works | 1,400 |
| 5 Science | 1,400 |
| 6 Technology | 1,400 |
| 1 Philosophy and psychology | 700 |
| 3 Social Science | 700 |
| 4 Language | 700 |
| 7 Arts & recreation | 700 |
| 8 Literature | 700 |
| 9 History | 700 |

## 4. 과제 유형 대응

원천마다 카테고리 체계가 달라 아래처럼 묶었다(`categories.py` `TASK_TYPE`).

| 과제 유형 | 원천 카테고리 |
|---|---|
| 수학 | routerbench:math, xroutebench:math, xroutebench:gsm8k, xroutebench:aime_2020_2024, llmrouterbench:livemathbench, llmrouterbench:aime |
| 코드 | routerbench:code, xroutebench:mbpp, xroutebench:human_eval, llmrouterbench:livecodebench |
| 지식 객관식 | routerbench:knowledge, xroutebench:mmlu, xroutebench:mmlu_pro, llmrouterbench:mmlupro |
| 상식·추론 | routerbench:commonsense, xroutebench:arc_challenge, xroutebench:hellaswag, xroutebench:openbook_qa, xroutebench:commonsense_qa |
| 독해 | xroutebench:squad, xroutebench:boolq |
| 전문가 과학·난제 | llmrouterbench:gpqa, llmrouterbench:hle, sprout:Idavidrein/gpqa/gpqa_extended |
| 사실 회상 | llmrouterbench:simpleqa |
| 열린 대화·지시 | routerbench:conversation, llmrouterbench:arenahard, sprout:openhermes/teknium |
