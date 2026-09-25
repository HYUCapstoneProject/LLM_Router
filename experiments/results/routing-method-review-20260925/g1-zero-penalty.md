# G1 비용 감점 없는 사후 진단 (λ=0)

Post-hoc diagnostic with no inference-time cost penalty. R3 training already includes cost; this does not remove cost from its learned objective. Mean observed benchmark quality is not difficulty classification accuracy. Source macro equally weights sources. No model or policy reselection.

원본 route_actions와 독립 NumPy raw-logit argmax가 모든 행에서 일치. 잠긴 선정 결과·정책 변경 없음.

## selection

| arm | λ=0 source-macro 관측 품질 | λ=0 관측 성공률 | 기존 기본예산 품질 | 기존 기본예산 상대비용 |
|---|---:|---:|---:|---:|
| r1-natural | 0.764339 | 0.751661 | 0.751785 | 0.426774 |
| r1-balanced | 0.763903 | 0.751607 | 0.747759 | 0.493867 |
| r3-natural | 0.701311 | 0.684837 | 0.701311 | 0.312073 |
| r3-balanced | 0.704135 | 0.687291 | 0.701984 | 0.256387 |

| source | task | n | r1-natural | r1-balanced | r3-natural | r3-balanced |
|---|---|---:|---:|---:|---:|---:|
| routerbench | ALL | 2060 | 0.791578 | 0.797767 | 0.746432 | 0.748252 |
| routerbench | gsm8k | 477 | 0.627358 | 0.633124 | 0.574948 | 0.572327 |
| routerbench | hellaswag | 625 | 0.902400 | 0.905600 | 0.888000 | 0.889600 |
| routerbench | mmlu_family | 871 | 0.804822 | 0.814007 | 0.741676 | 0.748565 |
| routerbench | mtbench | 5 | 0.480000 | 0.480000 | 0.480000 | 0.480000 |
| routerbench | winogrande | 82 | 0.780488 | 0.780488 | 0.731707 | 0.707317 |
| xroutebench | ALL | 254 | 0.737101 | 0.730038 | 0.656190 | 0.660017 |
| xroutebench | aime | 5 | 0.200000 | 0.200000 | 0.200000 | 0.200000 |
| xroutebench | boolq | 22 | 0.818182 | 0.818182 | 0.772727 | 0.772727 |
| xroutebench | commonsense_qa | 37 | 0.756757 | 0.729730 | 0.648649 | 0.675676 |
| xroutebench | gsm8k | 18 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| xroutebench | hellaswag | 30 | 0.833333 | 0.833333 | 0.533333 | 0.533333 |
| xroutebench | math_competition | 40 | 0.675000 | 0.650000 | 0.650000 | 0.650000 |
| xroutebench | mmlu_family | 32 | 0.906250 | 0.937500 | 0.843750 | 0.843750 |
| xroutebench | openbook_qa | 34 | 0.941176 | 0.941176 | 0.852941 | 0.852941 |
| xroutebench | squad | 36 | 0.256214 | 0.234158 | 0.240894 | 0.240122 |

Balanced − natural:
- r1: λ=0 품질 -0.000437; 기본예산 품질 -0.004027; 기본예산 상대비용 +0.067093.
- r3: λ=0 품질 +0.002824; 기본예산 품질 +0.000673; 기본예산 상대비용 -0.055686.

## task_holdout

| arm | λ=0 source-macro 관측 품질 | λ=0 관측 성공률 | 기존 기본예산 품질 | 기존 기본예산 상대비용 |
|---|---:|---:|---:|---:|
| r1-natural | 0.685145 | 0.685145 | 0.652622 | 0.412801 |
| r1-balanced | 0.699084 | 0.699084 | 0.669544 | 0.567213 |
| r3-natural | 0.684976 | 0.684976 | 0.684976 | 0.374278 |
| r3-balanced | 0.683398 | 0.683398 | 0.682972 | 0.359348 |

| source | task | n | r1-natural | r1-balanced | r3-natural | r3-balanced |
|---|---|---:|---:|---:|---:|---:|
| routerbench | ALL | 1173 | 0.868713 | 0.893436 | 0.843137 | 0.843137 |
| routerbench | arc_challenge | 910 | 0.948352 | 0.946154 | 0.895604 | 0.898901 |
| routerbench | mbpp | 263 | 0.593156 | 0.711027 | 0.661597 | 0.650190 |
| xroutebench | ALL | 634 | 0.501577 | 0.504732 | 0.526814 | 0.523659 |
| xroutebench | arc_challenge | 311 | 0.935691 | 0.945338 | 0.964630 | 0.958199 |
| xroutebench | human_eval | 90 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| xroutebench | mbpp | 233 | 0.115880 | 0.111588 | 0.145923 | 0.145923 |

Balanced − natural:
- r1: λ=0 품질 +0.013939; 기본예산 품질 +0.016923; 기본예산 상대비용 +0.154413.
- r3: λ=0 품질 -0.001577; 기본예산 품질 -0.002004; 기본예산 상대비용 -0.014930.

