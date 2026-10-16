# 캐시 실험 결과 초안 (cache-experiment-v1 / run001)

`06_report.ipynb`가 봉인된 산출물로 만든 **초안**이다. 해석과 결론은 사람이 검토해 채운다.

## 준비

- pool 5335행, D1 pool 5035행, D2 원질문 300그룹
- 운영 패리티: {"probes": 73, "top_candidate_mismatches": 0, "max_cosine_diff": 7.923801433884492e-07, "max_rerank_diff": 2.6226043701171875e-06, "cosine_decision_mismatches": 0, "rerank_decision_mismatches": 0, "passed": true}
- 채점기 자기 일치: {"rows": 4050, "decided": 4042, "unknown": 8, "agreement_on_decided": 0.9992577931716972, "disagreements": 3}
- 추론 장치: {"embed": "cuda", "rerank": "cuda", "router": "cuda", "router_package": "/content/drive/MyDrive/LLM_Router/transfer-experiments/inputs/experiment-A/models/r0"}; 임계값 ±1e-4 안의 판정 수(D1 C2·C3 합): 46

## D1 (운영 설정, seed 평균)

| repeat_fraction | arm | hit_rate_mean | exact_hit_rate_mean | semantic_hit_rate_mean | false_reuse_among_hits_mean | false_reuse_per_request_mean | unknown_among_hits_mean | call_reduction_mean | llm_cost_reduction_mean | quality_change_conservative_mean | quality_change_optimistic_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | C0 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0000 | C1 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0000 | C2 | 0.1840 | 0.0000 | 0.1840 | 1.0000 | 0.1840 | 0.3757 | 0.1840 | 0.1644 | -0.1135 | -0.0444 |
| 0.0000 | C3 | 0.0326 | 0.0000 | 0.0326 | 1.0000 | 0.0326 | 0.3968 | 0.0326 | 0.0404 | -0.0198 | -0.0068 |
| 0.0000 | C_cal | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.1000 | C0 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.1000 | C1 | 0.1000 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.1077 | 0.0000 | 0.0000 |
| 0.1000 | C2 | 0.2607 | 0.0862 | 0.1746 | 0.6695 | 0.1746 | 0.2534 | 0.2607 | 0.2456 | -0.1072 | -0.0412 |
| 0.1000 | C3 | 0.1285 | 0.0976 | 0.0309 | 0.2405 | 0.0309 | 0.0934 | 0.1285 | 0.1425 | -0.0183 | -0.0063 |
| 0.1000 | C_cal | 0.1000 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.1077 | 0.0000 | 0.0000 |
| 0.3000 | C0 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.3000 | C1 | 0.3000 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 0.2862 | 0.0000 | 0.0000 |
| 0.3000 | C2 | 0.4160 | 0.2637 | 0.1522 | 0.3659 | 0.1522 | 0.1375 | 0.4160 | 0.3830 | -0.0945 | -0.0374 |
| 0.3000 | C3 | 0.3201 | 0.2936 | 0.0265 | 0.0828 | 0.0265 | 0.0341 | 0.3201 | 0.3105 | -0.0147 | -0.0038 |
| 0.3000 | C_cal | 0.3000 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 0.2862 | 0.0000 | 0.0000 |
| 0.5000 | C0 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 | nan | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.5000 | C1 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4859 | 0.0000 | 0.0000 |
| 0.5000 | C2 | 0.5741 | 0.4438 | 0.1303 | 0.2269 | 0.1303 | 0.0774 | 0.5741 | 0.5495 | -0.0821 | -0.0377 |
| 0.5000 | C3 | 0.5137 | 0.4894 | 0.0242 | 0.0472 | 0.0242 | 0.0188 | 0.5137 | 0.5021 | -0.0129 | -0.0032 |
| 0.5000 | C_cal | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4859 | 0.0000 | 0.0000 |

## D2 보정 (calibration 100그룹)

- 상태: `no_policy_meets_limit_fallback_exact_only` (상한 false reuse among hits ≤ 0.01)
- 봉인 정책: `null` → 운영 형태 `{"exact": true, "semantic": false}`
- 상한을 만족하는 의미 캐시 정책이 없어 C_cal은 exact-only다. D1의 C_cal은 C1과 같은 정책이다.
- 격자 3127개 설정 중 false reuse among hits 최저 설정:

| kind | k | cosine | rerank_similarity | rerank_score | hits | false_reuse | false_reuse_among_hits | reusable_hits | paraphrase_hit_rate | hard_negative_hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| rerank | 1 | 0.96 | 0.96 | 0.9 | 64 | 17 | 0.265625 | 37 | 0.47 | 0.17 |

## D2 test

판정 방식: single_llm — **LLM 판정 1개 기준. 사람 검증된 안전성으로 주장하지 않는다.**

```json
{
 "C2": {
  "spec": {
   "kind": "cosine",
   "k": 5,
   "cosine": 0.8
  },
  "overall": {
   "items": 400,
   "hits": 396,
   "false_reuse": 197,
   "false_reuse_among_hits": 0.49747474747474746,
   "false_reuse_ci95_low": 0.4471452176808231,
   "false_reuse_ci95_high": 0.5478422886418057,
   "reusable_items": 137,
   "reusable_hits": 137,
   "reusable_recall": 1.0,
   "wrong_answers_among_hits": 224,
   "unknown_answers_among_hits": 35,
   "answer_error_among_hits_conservative": 0.6540404040404041,
   "answer_error_among_hits_optimistic": 0.5656565656565656,
   "hard_negative_hit_rate": 0.98,
   "paraphrase_hit_rate": 1.0
  },
  "excluding_flagged_groups": {
   "groups_excluded": [
    "d2-055",
    "d2-114"
   ],
   "items": 396,
   "hits": 392,
   "false_reuse": 195,
   "false_reuse_among_hits": 0.49744897959183676,
   "false_reuse_ci95_low": 0.44685884727568326,
   "false_reuse_ci95_high": 0.5480779078208937,
   "reusable_items": 137,
   "reusable_hits": 137,
   "reusable_recall": 1.0,
   "wrong_answers_among_hits": 220,
   "unknown_answers_among_hits": 35,
   "answer_error_among_hits_conservative": 0.6505102040816326,
   "answer_error_among_hits_optimistic": 0.5612244897959183,
   "hard_negative_hit_rate": 0.9797979797979798,
   "paraphrase_hit_rate": 1.0
  },
  "by_stratum": {
   "rb_math": {
    "items": 68,
    "hits": 68,
    "false_reuse": 34,
    "false_reuse_among_hits": 0.5,
    "false_reuse_ci95_low": 0.37620358089087436,
    "false_reuse_ci95_high": 0.6237964191091256,
    "reusable_items": 20,
    "reusable_hits": 20,
    "reusable_recall": 1.0,
    "wrong_answers_among_hits": 45,
    "unknown_answers_among_hits": 3,
    "answer_error_among_hits_conservative": 0.7058823529411765,
    "answer_error_among_hits_optimistic": 0.6617647058823529,
    "hard_negative_hit_rate": 1.0,
    "paraphrase_hit_rate": 1.0
   },
   "rb_mcq": {
    "items": 172,
    "hits": 168,
    "false_reuse": 82,
    "false_reuse_among_hits": 0.4880952380952381,
    "false_reuse_ci95_low": 0.41033958833637924,
    "false_reuse_ci95_high": 0.5662782752472473,
    "reusable_items": 58,
    "reusable_hits": 58,
    "reusable_recall": 1.0,
    "wrong_answers_among_hits": 89,
    "unknown_answers_among_hits": 21,
    "answer_error_among_hits_conservative": 0.6547619047619048,
    "answer_error_among_hits_optimistic": 0.5297619047619048,
    "hard_negative_hit_rate": 0.9534883720930233,
    "paraphrase_hit_rate": 1.0
   },
   "rb_open": {
    "items": 28,
    "hits": 28,
    "false_reuse": 14,
    "false_reuse_among_hits": 0.5,
    "false_reuse_ci95_low": 0.306470961532318,
    "false_reuse_ci95_high": 0.693529038467682,
    "reusable_items": 7,
    "reusable_hits": 7,
    "reusable_recall": 1.0,
    "wrong_answers_among_hits": 20,
    "unknown_answers_among_hits": 1,
    "answer_error_among_hits_conservative": 0.75,
    "answer_error_among_hits_optimistic": 0.7142857142857143,
    "hard_negative_hit_rate": 1.0,
    "paraphrase_hit_rate": 1.0
   },
   "xrb_mcq": {
    "items": 68,
    "hits": 68,
    "false_reuse": 35,
    "false_reuse_among_hits": 0.5147058823529411,
    "false_reuse_ci95_low": 0.3902981879797207,
    "false_reuse_ci95_high": 0.6377919077023989,
    "reusable_items": 28,
    "reusable_hits": 28,
    "reusable_recall": 1.0,
    "wrong_answers_among_hits": 31,
    "unknown_answers_among_hits": 9,
    "answer_error_among_hits_conservative": 0.5882352941176471,
    "answer_error_among_hits_optimistic": 0.45588235294117646,
    "hard_negative_hit_rate": 1.0,
    "paraphrase_hit_rate": 1.0
   },
   "xrb_numeric": {
    "items": 36,
    "hits": 36,
    "false_reuse": 18,
    "false_reuse_among_hits": 0.5,
    "false_reuse_ci95_low": 0.3292217994577103,
    "false_reuse_ci95_high": 0.6707782005422896,
    "reusable_items": 10,
    "reusable_hits": 10,
    "reusable_recall": 1.0,
    "wrong_answers_among_hits": 26,
    "unknown_answers_among_hits": 0,
    "answer_error_among_hits_conservative": 0.7222222222222222,
    "answer_error_among_hits_optimistic": 0.7222222222222222,
    "hard_negative_hit_rate": 1.0,
    "paraphrase_hit_rate": 1.0
   },
   "xrb_open": {
    "items": 28,
    "hits": 28,
    "false_reuse": 14,
    "false_reuse_among_hits": 0.5,
    "false_reuse_ci95_low": 0.306470961532318,
    "false_reuse_ci95_high": 0.693529038467682,
    "reusable_items": 14,
    "reusable_hits": 14,
    "reusable_recall": 1.0,
    "wrong_answers_among_hits": 13,
    "unknown_answers_among_hits": 1,
    "answer_error_among_hits_conservative": 0.5,
    "answer_error_among_hits_optimistic": 0.4642857142857143,
    "hard_negative_hit_rate": 1.0,
    "paraphrase_hit_rate": 1.0
   }
  },
  "by_negative_type": {
   "code_language": {
    "items": 8,
    "hits": 8,
    "false_reuse": 8,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.6305833524471807,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 1,
    "reusable_hits": 1,
    "reusable_recall": 1.0,
    "wrong_answers_among_hits": 7,
    "unknown_answers_among_hits": 0,
    "answer_error_among_hits_conservative": 0.875,
    "answer_error_among_hits_optimistic": 0.875,
    "hard_negative_hit_rate": 1.0,
    "paraphrase_hit_rate": null
   },
   "entity_swap": {
    "items": 33,
    "hits": 33,
    "false_reuse": 33,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.8942371899254207,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 28,
    "unknown_answers_among_hits": 5,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.8484848484848485,
    "hard_negative_hit_rate": 1.0,
    "paraphrase_hit_rate": null
   },
   "negation": {
    "items": 24,
    "hits": 23,
    "false_reuse": 23,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.8518148710847756,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 20,
    "unknown_answers_among_hits": 3,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.8695652173913043,
    "hard_negative_hit_rate": 0.9583333333333334,
    "paraphrase_hit_rate": null
   },
   "number_unit": {
    "items": 41,
    "hits": 40,
    "false_reuse": 40,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.9119026971211976,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 33,
    "unknown_answers_among_hits": 7,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.825,
    "hard_negative_hit_rate": 0.975609756097561,
    "paraphrase_hit_rate": null
   },
   "order": {
    "items": 41,
    "hits": 39,
    "false_reuse": 39,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.9097488990396657,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 33,
    "unknown_answers_among_hits": 6,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.8461538461538461,
    "hard_negative_hit_rate": 0.9512195121951219,
    "paraphrase_hit_rate": null
   },
   "output_format": {
    "items": 16,
    "hits": 16,
    "false_reuse": 16,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.7940927857921772,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 16,
    "unknown_answers_among_hits": 0,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 1.0,
    "hard_negative_hit_rate": 1.0,
    "paraphrase_hit_rate": null
   },
   "time": {
    "items": 37,
    "hits": 37,
    "false_reuse": 37,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.9051094125850101,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 31,
    "unknown_answers_among_hits": 6,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.8378378378378378,
    "hard_negative_hit_rate": 1.0,
    "paraphrase_hit_rate": null
   }
  }
 },
 "C3": {
  "spec": {
   "kind": "rerank",
   "k": 5,
   "cosine": 0.8,
   "rerank_similarity": 0.8774,
   "rerank_score": 0.03581769
  },
  "overall": {
   "items": 400,
   "hits": 322,
   "false_reuse": 146,
   "false_reuse_among_hits": 0.453416149068323,
   "false_reuse_ci95_low": 0.3981378340325532,
   "false_reuse_ci95_high": 0.5095592048472034,
   "reusable_items": 137,
   "reusable_hits": 122,
   "reusable_recall": 0.8905109489051095,
   "wrong_answers_among_hits": 169,
   "unknown_answers_among_hits": 30,
   "answer_error_among_hits_conservative": 0.6180124223602484,
   "answer_error_among_hits_optimistic": 0.5248447204968945,
   "hard_negative_hit_rate": 0.72,
   "paraphrase_hit_rate": 0.89
  },
  "excluding_flagged_groups": {
   "groups_excluded": [
    "d2-055",
    "d2-114"
   ],
   "items": 396,
   "hits": 318,
   "false_reuse": 144,
   "false_reuse_among_hits": 0.4528301886792453,
   "false_reuse_ci95_low": 0.397212653576576,
   "false_reuse_ci95_high": 0.5093345046516031,
   "reusable_items": 137,
   "reusable_hits": 122,
   "reusable_recall": 0.8905109489051095,
   "wrong_answers_among_hits": 165,
   "unknown_answers_among_hits": 30,
   "answer_error_among_hits_conservative": 0.6132075471698113,
   "answer_error_among_hits_optimistic": 0.5188679245283019,
   "hard_negative_hit_rate": 0.7171717171717171,
   "paraphrase_hit_rate": 0.8888888888888888
  },
  "by_stratum": {
   "rb_math": {
    "items": 68,
    "hits": 62,
    "false_reuse": 28,
    "false_reuse_among_hits": 0.45161290322580644,
    "false_reuse_ci95_low": 0.3248059720229999,
    "false_reuse_ci95_high": 0.5831963456545707,
    "reusable_items": 20,
    "reusable_hits": 20,
    "reusable_recall": 1.0,
    "wrong_answers_among_hits": 40,
    "unknown_answers_among_hits": 2,
    "answer_error_among_hits_conservative": 0.6774193548387096,
    "answer_error_among_hits_optimistic": 0.6451612903225806,
    "hard_negative_hit_rate": 0.8235294117647058,
    "paraphrase_hit_rate": 1.0
   },
   "rb_mcq": {
    "items": 172,
    "hits": 137,
    "false_reuse": 61,
    "false_reuse_among_hits": 0.44525547445255476,
    "false_reuse_ci95_low": 0.36039534862555783,
    "false_reuse_ci95_high": 0.5325337628121851,
    "reusable_items": 58,
    "reusable_hits": 53,
    "reusable_recall": 0.9137931034482759,
    "wrong_answers_among_hits": 67,
    "unknown_answers_among_hits": 17,
    "answer_error_among_hits_conservative": 0.6131386861313869,
    "answer_error_among_hits_optimistic": 0.48905109489051096,
    "hard_negative_hit_rate": 0.7093023255813954,
    "paraphrase_hit_rate": 0.8837209302325582
   },
   "rb_open": {
    "items": 28,
    "hits": 22,
    "false_reuse": 8,
    "false_reuse_among_hits": 0.36363636363636365,
    "false_reuse_ci95_low": 0.1719794376463324,
    "false_reuse_ci95_high": 0.5934233311743533,
    "reusable_items": 7,
    "reusable_hits": 7,
    "reusable_recall": 1.0,
    "wrong_answers_among_hits": 13,
    "unknown_answers_among_hits": 2,
    "answer_error_among_hits_conservative": 0.6818181818181818,
    "answer_error_among_hits_optimistic": 0.5909090909090909,
    "hard_negative_hit_rate": 0.5714285714285714,
    "paraphrase_hit_rate": 1.0
   },
   "xrb_mcq": {
    "items": 68,
    "hits": 49,
    "false_reuse": 23,
    "false_reuse_among_hits": 0.46938775510204084,
    "false_reuse_ci95_low": 0.32533872511655376,
    "false_reuse_ci95_high": 0.6172692650662404,
    "reusable_items": 28,
    "reusable_hits": 22,
    "reusable_recall": 0.7857142857142857,
    "wrong_answers_among_hits": 21,
    "unknown_answers_among_hits": 6,
    "answer_error_among_hits_conservative": 0.5510204081632653,
    "answer_error_among_hits_optimistic": 0.42857142857142855,
    "hard_negative_hit_rate": 0.6470588235294118,
    "paraphrase_hit_rate": 0.7941176470588235
   },
   "xrb_numeric": {
    "items": 36,
    "hits": 30,
    "false_reuse": 15,
    "false_reuse_among_hits": 0.5,
    "false_reuse_ci95_low": 0.31297028586802744,
    "false_reuse_ci95_high": 0.6870297141319724,
    "reusable_items": 10,
    "reusable_hits": 9,
    "reusable_recall": 0.9,
    "wrong_answers_among_hits": 18,
    "unknown_answers_among_hits": 2,
    "answer_error_among_hits_conservative": 0.6666666666666666,
    "answer_error_among_hits_optimistic": 0.6,
    "hard_negative_hit_rate": 0.7777777777777778,
    "paraphrase_hit_rate": 0.8888888888888888
   },
   "xrb_open": {
    "items": 28,
    "hits": 22,
    "false_reuse": 11,
    "false_reuse_among_hits": 0.5,
    "false_reuse_ci95_low": 0.2822105209380929,
    "false_reuse_ci95_high": 0.717789479061907,
    "reusable_items": 14,
    "reusable_hits": 11,
    "reusable_recall": 0.7857142857142857,
    "wrong_answers_among_hits": 10,
    "unknown_answers_among_hits": 1,
    "answer_error_among_hits_conservative": 0.5,
    "answer_error_among_hits_optimistic": 0.45454545454545453,
    "hard_negative_hit_rate": 0.7857142857142857,
    "paraphrase_hit_rate": 0.7857142857142857
   }
  },
  "by_negative_type": {
   "code_language": {
    "items": 8,
    "hits": 4,
    "false_reuse": 4,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.3976353643835253,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 1,
    "reusable_hits": 1,
    "reusable_recall": 1.0,
    "wrong_answers_among_hits": 3,
    "unknown_answers_among_hits": 0,
    "answer_error_among_hits_conservative": 0.75,
    "answer_error_among_hits_optimistic": 0.75,
    "hard_negative_hit_rate": 0.5,
    "paraphrase_hit_rate": null
   },
   "entity_swap": {
    "items": 33,
    "hits": 19,
    "false_reuse": 19,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.8235330881930346,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 15,
    "unknown_answers_among_hits": 4,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.7894736842105263,
    "hard_negative_hit_rate": 0.5757575757575758,
    "paraphrase_hit_rate": null
   },
   "negation": {
    "items": 24,
    "hits": 13,
    "false_reuse": 13,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.752947361999529,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 12,
    "unknown_answers_among_hits": 1,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.9230769230769231,
    "hard_negative_hit_rate": 0.5416666666666666,
    "paraphrase_hit_rate": null
   },
   "number_unit": {
    "items": 41,
    "hits": 26,
    "false_reuse": 26,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.8677253955022459,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 21,
    "unknown_answers_among_hits": 5,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.8076923076923077,
    "hard_negative_hit_rate": 0.6341463414634146,
    "paraphrase_hit_rate": null
   },
   "order": {
    "items": 41,
    "hits": 35,
    "false_reuse": 35,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.8999675644278949,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 30,
    "unknown_answers_among_hits": 5,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.8571428571428571,
    "hard_negative_hit_rate": 0.8536585365853658,
    "paraphrase_hit_rate": null
   },
   "output_format": {
    "items": 16,
    "hits": 13,
    "false_reuse": 13,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.752947361999529,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 12,
    "unknown_answers_among_hits": 1,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.9230769230769231,
    "hard_negative_hit_rate": 0.8125,
    "paraphrase_hit_rate": null
   },
   "time": {
    "items": 37,
    "hits": 34,
    "false_reuse": 34,
    "false_reuse_among_hits": 1.0,
    "false_reuse_ci95_low": 0.8971820757409874,
    "false_reuse_ci95_high": 1.0,
    "reusable_items": 0,
    "reusable_hits": 0,
    "reusable_recall": null,
    "wrong_answers_among_hits": 28,
    "unknown_answers_among_hits": 6,
    "answer_error_among_hits_conservative": 1.0,
    "answer_error_among_hits_optimistic": 0.8235294117647058,
    "hard_negative_hit_rate": 0.918918918918919,
    "paraphrase_hit_rate": null
   }
  }
 },
 "C1": {
  "note": "exact-only; a variant hits only if identical after whitespace normalization",
  "exact_hits": 0
 }
}
```

## 지연 (로컬 실측)

| stage | n | p50_ms | p95_ms | mean_ms | max_ms | max_rss_mb | threads | device | cache_size | ef_search |
|---|---|---|---|---|---|---|---|---|---|---|
| embed | 500 | 174.7236 | 415.2573 | 215.7744 | 640.7040 | 1933.2344 | 2.0000 | cpu | nan | nan |
| rerank_batch5 | 500 | 410.1371 | 592.2934 | 416.8261 | 758.9299 | 1933.2344 | 2.0000 | cpu | nan | nan |
| route | 500 | 73.7114 | 191.5727 | 97.3693 | 457.3974 | 1933.2344 | 2.0000 | None | nan | nan |
| pg_exact_lookup | 500 | 0.1339 | 0.2068 | 0.1430 | 0.3900 | 1933.2344 | nan | nan | 500.0000 | 40 |
| pg_vector_search | 500 | 3.3705 | 5.2889 | 3.6184 | 6.7637 | 1933.2344 | nan | nan | 500.0000 | 40 |
| pg_exact_lookup | 500 | 0.1325 | 0.1725 | 0.1360 | 0.4361 | 1933.2344 | nan | nan | 2000.0000 | 40 |
| pg_vector_search | 500 | 1.7177 | 2.8276 | 1.8499 | 3.5238 | 1933.2344 | nan | nan | 2000.0000 | 40 |
| pg_exact_lookup | 500 | 0.1367 | 0.1804 | 0.1384 | 0.2425 | 1933.2344 | nan | nan | 4000.0000 | 40 |
| pg_vector_search | 500 | 3.9125 | 6.6342 | 3.7279 | 8.4461 | 1933.2344 | nan | nan | 4000.0000 | 40 |

연산 단가: None USD/h (None, None)

## 그림

- `report/figures/d1-call-reduction.png`
- `report/figures/safety-vs-savings.png`
- `report/figures/cumulative.png`
