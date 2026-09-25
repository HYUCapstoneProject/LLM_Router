# R0: train-only majority 기준선 감사

판단: 특히 xRouteBench에서 출처·카테고리별 다수 라벨만으로 R0와 거의 같은 정확도를 얻었다. 단순한 카테고리 정보가 높은 난이도 정확도를 설명할 수 있다는 진단 근거다. R0가 실제로 카테고리 shortcut에 의존한다는 인과 증명은 아니다.

학습 21,406행의 라벨 빈도로만 예측을 고정하고 D_dev 3,581행에 적용했다. 모델 재학습·추론·test 접근·정책 재선택은 없다. 동률은 easy→medium→hard 순서다. 미지의 (source, category)는 해당 source의 train 다수 라벨, 미지의 source는 global train 다수 라벨로 fallback한다. 실제 D_dev에서는 모두 알려진 조합이라 fallback 0건이었다.

| 구간 | 방법 | accuracy | macro F1 | balanced accuracy |
|---|---|---:|---:|---:|
| 전체 | source_majority | 0.493717 | 0.220353 | 0.333333 |
| 전체 | source_category_majority | 0.553756 | 0.425856 | 0.428596 |
| 전체 | r0_raw_argmax | 0.605417 | 0.519767 | 0.507923 |
| routerbench | source_majority | 0.480782 | 0.216454 | 0.333333 |
| routerbench | source_category_majority | 0.530109 | 0.343992 | 0.382631 |
| routerbench | r0_raw_argmax | 0.588405 | 0.461652 | 0.462559 |
| xroutebench | source_majority | 0.581699 | 0.245179 | 0.333333 |
| xroutebench | source_category_majority | 0.714597 | 0.598579 | 0.574953 |
| xroutebench | r0_raw_argmax | 0.721133 | 0.590526 | 0.596013 |

- 출처별 majority는 두 source 모두 easy를 고른다. 출처·카테고리 majority 대비 R0의 pooled 정확도 개선은 +5.166pp(+185문항), RouterBench +5.830pp(+182문항), xRouteBench +0.654pp(+3문항)다.
- xRouteBench에서 category baseline macro F1 0.598579는 R0 0.590526보다 약 0.805pp 높다. 예측 일치율도 86.93%(399/459)다. medium recall은 baseline 17/92=18.48%, R0 12/92=13.04%다.
- 전체 예측 일치율은 69.25%; baseline만 정답 358건, R0만 정답 543건이다. 따라서 R0 전체 성능을 카테고리 majority 하나로 완전히 설명할 수는 없다.
- RouterBench knowledge에서는 R0가 baseline보다 +10.57pp(n=1,258), math +4.48pp(n=670)로 개선된다. 반면 xRouteBench SQuAD(n=39)·AIME(n=11)는 baseline과 R0 모두 100%다. gsm8k(n=47)에서는 baseline 78.72% > R0 70.21%다. 작은 표본의 큰 차이는 과해석하지 않는다.

모든 수치는 R0 raw argmax의 난이도 분류만 비교한다. 비용 정책 적용 결과와 R1/R2 success/ranking 출력은 포함하지 않았다. 서로 다른 공개 모델 풀과 정답 기준 때문에 source 간 차이를 단순한 학습 비율의 인과 효과로 주장하지 않는다. 추가적인 카테고리 제거/교란, unseen-category 평가가 없으므로 shortcut 사용 여부는 미확정이다.

macro F1은 항상 easy/medium/hard 세 라벨을 평균한다. balanced accuracy는 각 평가 slice에 실제 존재하는 라벨의 recall 평균이다. 카테고리별 1–2개 라벨만 있는 구간은 support와 함께 봐야 한다. JSON에는 고정 3클래스 macro recall도 별도 보관했다. train/dev 무결성은 원본 manifest chain으로 검증했고 train source/category/label은 원본 R0 base training과 일치했다. sample ID·duplicate group도 train/dev 간 겹치지 않는다.

산출물: `/private/tmp/llm-router-r0-category-baselines.json`(원본 경로·SHA·train prior·모든 source/category 지표), `.parquet`(행별 정답/세 예측), `.py`(재현 스크립트).
