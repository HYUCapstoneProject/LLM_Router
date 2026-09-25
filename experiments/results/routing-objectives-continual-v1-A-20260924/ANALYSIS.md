# 실제 R0/R1/R2 비교 결과 해석

`REPORT.md`와 `report.json`은 `render-routing-objective-report.py`가 원본 검증 산출물과 validation lock을 재검산해 생성했다. 이 문서는 그 결과의 해석과 보관 범위를 보충한다. 추가 학습, 정책 재선택, sealed 평가, 운영 활성화는 수행하지 않았다.

## 선정 결과와 해석

선정 지표인 D_dev source-macro 비용–품질 nAUC는 **R1 0.863179 > R2 0.833874 > R0 0.166115**다. R1은 R2보다 0.029305 높고, RouterBench와 xRouteBench의 nAUC에서도 각각 0.045845, 0.012765 높다. 기존 validation lock의 R1 승자를 재확인한 결과다.

이 수치는 성공 확률 86.3%나 Claude 비용 절감률이 아니다. 각 source의 always-low/high 기준으로 정규화한 품질 회복량을 정해진 상대 비용 budget grid에서 적분한 값이다. RouterBench 3,122행과 xRouteBench 459행을 각각 평가하고 두 source에 같은 비중을 준다. 더 작은 xRouteBench의 표본 불확실성도 별도로 고려해야 한다.

R1이 모든 비용 조건에서 R2보다 좋은 것은 아니다. 미리 정한 기본 상대 비용 budget **0.50**의 동결 정책을 다시 적용하면 다음과 같다.

| source | 방법 | λ | 관측 평균 품질 | 상대 실제 비용 | low / medium / high 선택 건수 |
|---|---|---:|---:|---:|---|
| RouterBench | R0 | 0 | 0.600993 | 0.076359 | 1,829 / 1,160 / 133 |
| RouterBench | R1 | 0.10 | 0.795996 | 0.494171 | 12 / 1,732 / 1,378 |
| RouterBench | R2 | 1.00 | 0.742008 | 0.135831 | 134 / 2,748 / 240 |
| xRouteBench | R0 | 0.003 | 0.675099 | 0.234265 | 336 / 29 / 94 |
| xRouteBench | R1 | 0.03 | 0.739370 | 0.459636 | 122 / 159 / 178 |
| xRouteBench | R2 | 0.10 | 0.744968 | 0.452686 | 146 / 129 / 184 |

- RouterBench의 기본 budget에서는 R1이 R2보다 높은 품질과 높은 비용을 함께 사용했다. 같은 상한 아래의 비교이며 지출 금액이 같은 비교는 아니다.
- xRouteBench의 기본 budget에서는 R2의 품질이 R1보다 0.005598 높고 비용도 소폭 낮다. 이 사실은 전체 budget 구간을 종합하는 nAUC의 R1 선정과 모순되지 않는다.
- RouterBench에서 R1의 low 선택은 12/3,122건으로 적지만, medium/high를 포함한 세 행동은 모두 실제로 사용됐다. 한 budget에서의 사용 비중을 전체 정책이나 실제 Claude 사용 비중으로 일반화할 수 없다.
- R0는 budget을 늘려도 높은 품질의 정책 점을 충분히 제공하지 못했다. 다만 이 관측만으로 T1 난이도 분류 정확도가 나쁘다고 단정하거나, 라벨 압축이 차이의 원인이라고 확정하지 않는다.

위 집계의 기계 판독 값과 선택 건수는 `analysis.json`에 보관한다. λ는 validation lock에서 읽은 값을 그대로 썼으며 다시 최적화하지 않았다.

## 세 단계와 실제 후보의 의미

R0/R1/R2 모두 **세 개의 출력과 low/medium/high 세 행동을 유지**한다. 서비스에서의 호환 투영은 `low → easy → haiku`, `medium → medium → sonnet`, `high → hard → opus`다. 학습 목표와 score의 의미가 바뀌는 것이며, R1이 이겼다고 두 단계 모델로 바뀌지 않는다.

그러나 이번 공개 벤치마크에서 실제로 평가한 모델은 아래와 같다. Claude 모델을 평가한 결과가 아니다.

| source | low | medium | high |
|---|---|---|---|
| RouterBench | `mistralai/mistral-7b-chat` | `zero-one-ai/Yi-34B-Chat` | `gpt-4-1106-preview` |
| xRouteBench | `gpt-oss-20b` | `gpt-oss-120b` | `cogito-v2-1-671b` |

후보 선택 규칙과 ID는 원본 `arms.json`의 train-only 설정이며 `evidence/arms.json`에 복사했다. R1은 이 공개 후보들의 success를 학습한 모델이다. 이를 현재 Claude 모델의 성공 확률로 바로 읽을 수 없다. 실제 서비스 모델 ID·가격/비용 버전·λ를 명시한 정책과 서비스 후보에 대한 평가가 필요하다. source별 λ를 평균해 서비스 λ로 바꾸지도 않는다.

## 실험과 검증 범위

- 동일한 train 21,406행, D_dev 3,581행, max length 512, 3 epochs, 2,007 optimizer steps, seed 42를 사용한 canonical 1회 비교다. B 실험이나 seed 반복 결과는 없다.
- R1/R2의 학습 시간과 R0의 기록 시간은 복구·실행 조건의 영향을 받을 수 있다. 기록된 시간 차이를 objective 자체의 계산 속도 차이나 총 GPU 과금 차이로 주장하지 않는다.
- nAUC 재계산, source별 곡선, 동결 budget 정책, 동일 입력 계약, manifest/file SHA와 승자 규칙을 generator가 검증했다. 두 PNG는 범례·축·수치가 겹치거나 잘리지 않는지 시각 확인했다.
- 선택에 사용한 D_dev 결과다. 통계적 유의성, 새 질문에 대한 일반화, seed 안정성, 실제 운영 품질을 확정하지 않는다. 이 보고서에는 paired bootstrap CI, D_gate, D_stream, D_retention_test, 고정 test, RouterArena 결과가 없다.
- R1의 운영 재학습은 관측 가능한 success target 및 미관측 처리, policy를 반영한 evaluator와 승격 gate 연결이 남아 있다. 체인 종료의 암묵적 `userSatisfied`를 후보 성공 정답으로 바꾸어 사용하지 않는다.

## 보관 근거와 재현

실행 입력은 코드 commit `85c422c9af4c51a51ba5eb45a0ec957b836e02d2`의 frozen bundle이다. 그 뒤 수정한 로컬 CLI·패키지·서빙 코드는 이번 학습 결과에 소급 적용되지 않는다.

비교 산출물 archive는 `gs://llm-router-mlserver/training-runs/objectives-continual-v1-A-20260924/comparison-results.tar`이며, 크기는 583,680 bytes, SHA-256은 `c3b844dbe003a4d209e7d304ca72ab3d38f7e780a85dd17d0de2d79f3aac67b1`이다. **이 tar에는 metrics, manifest, scores, curve, lock 등이 들어 있으며 model weights는 포함되지 않는다.** 보고서의 모델 크기는 run manifest의 기록이고, 이 보고서 생성 과정에서 실제 가중치를 로드하거나 검증하지 않았다.

상위 작업은 GCS receipt 15개의 크기·CRC32C·generation을 독립 확인한 뒤 archive를 내려받았다. 이 로컬 보고서 작업은 archive SHA 및 생성기에 필요한 로컬 파일을 검증했다. 상세 출처와 receipt 요약은 `evidence/transfer-summary.json`에 남겼다.

**개인 Drive 보존은 원본 전송 보고서에서 검증 완료로 기록돼 있다.** 실제 필드는 `personal_drive.status=verified_after_flush_and_remount`이며, Colab에서 Drive를 flush하고 다시 mount한 뒤 15개 파일의 크기·SHA-256을 확인한 receipt가 있다. 보존 경로는 `/content/drive/MyDrive/LLM_Router/routing-objectives-continual-v1-A-20260924`이고 `staging_errors`는 비어 있다. 이 receipt의 파일별 크기·SHA-256이 GCS receipt 15개와 일치함도 로컬에서 대조했다. 원본 `transfer-report.json`의 SHA-256은 `dc75526ee15908bb567bfe794d299f534a727e98cb3aec6776d364f06974ae3c`이다.

표준 `run.json`에 실제 보존 receipt와 운영 미활성화 상태를 기록하고, 이 기록을 `--run-record`로 전달해 `REPORT.md`와 `report.json`을 다시 생성했다. 생성기는 원본 score·curve·선택 lock을 다시 검증했으며 수치와 두 PNG의 SHA는 그대로다. 보고서의 보관 표는 외부 run 기록을 인용한 것이고, 생성기 자체가 원격 Drive/GCS를 다시 조회했다는 뜻은 아니다.

원본 transfer report는 `production_activated=false`를 기록한다. GCS·Drive 학습 백업은 운영 package 게시·CURRENT 갱신·serving 활성화와 별개다.

재현 명령은 아래와 같다. `comparison`은 위 archive를 푼 디렉터리이며 출력 디렉터리는 새 경로여야 한다.

```sh
PYTHONPATH=/private/tmp/llm-router-report-deps .venv/bin/python \
  ml/scripts/render-routing-objective-report.py \
  --runs-dir /private/tmp/llm-router-objective-results-local/comparison \
  --validation-lock /private/tmp/llm-router-objective-results-local/comparison/validation-lock.json \
  --run-record experiments/results/routing-objectives-continual-v1-A-20260924/run.json \
  --output-dir /path/to/new-report
```
