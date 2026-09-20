# ModernBERT continual retraining strategy

> 2026-09-20 실험 명세 추가: [offline escalation 실험](escalation-offline-experiment-plan.md)은
> 아래의 3-success cycle·anchor·replay·5-major rebase 정책을 재사용한다. 연구용 승격 gate는
> 별도 내부 검증 데이터로 분리해야 하며 RouterArena를 반복 승격 의사결정에 사용하지 않는다.
> 이는 현재 배포 gate가 변경됐다는 뜻이 아니다. strict upward feedback에서는 final easy가
> 나오지 않으므로 label당 32행 replay의 실제 가능한 상한은 medium/hard 합계 64행이다.

> 2026-09-16 통합 주석: continual retraining과 자동 승격은 ModernBERT만 대상으로 한다.
> 다른 모델군은 Colab notebook의 offline 비교 결과와 차기 권고에만 사용하며 KFP/Vertex에
> 올리지 않는다. ModernBERT 운영 초기학습도 Colab에서 수행하고, Vertex T4 학습은 이
> 문서의 feedback 재학습부터 사용한다. 통합 데이터로 전환할 때는
> base dataset·label·training config의 새 version을 만들되 ModernBERT package-v1과
> `models/router-modernbert/CURRENT.json` 경계를 유지하고 기존 artifact를 덮어쓰지 않는다.

상태: 구현 기준

기준일: 2026-09-12

## 버전과 학습 기준

현재 연구 단계에서는 `POST /escalate`로 제출되어 pipeline 승격까지 성공한 HTTP 호출 수가
유일한 version trigger다. 1건 이상의 batch를 허용하며 event 수, accepted event 수, 기간,
분포 같은 metadata는 version trigger에 사용하지 않는다.

```text
anchor 1.0
  call 1: anchor 1.0 + cycle B1 + compact replay + base rehearsal -> 1.1
  call 2: anchor 1.0 + cycle B1+B2 + compact replay + base rehearsal -> 1.2
  call 3: anchor 1.0 + cycle B1+B2+B3 + compact replay + base rehearsal -> 2.0
  new anchor: 2.0
```

minor가 직전 minor에서 연속 학습하지 않고 같은 `.0` anchor에서 cycle 누적 데이터를 다시
학습하는 이유는 호출 순서에 따른 drift와 catastrophic forgetting을 줄이기 위해서다. 첫 두
호출은 `minor_fine_tune`, 세 번째 호출은 `major_consolidation`으로 기록하지만 현재는 같은
deploy gate를 사용한다.

major consolidation이 5회 누적되는 `.0` 호출은 `pristine_rebase`가 된다. 이때만 승인된
anchor weight를 사용하지 않고 pinned `answerdotai/ModernBERT-base`에서 compact replay와
현재 cycle로 다시 학습해 장기적인 weight drift를 끊는다.

## 데이터 경로

Spring은 raw event 전체를 `feedback-inline-v1` JSON으로 FastAPI에 보낸다. FastAPI가
schema와 의미를 검증하고 ML용 Parquet, checksum과 `feedback-export-v0.1` manifest를
만든 뒤 다음 두 파생 artifact를 별도 GCS prefix에 immutable object로 기록한다.

- training feedback: 현재 3-call cycle의 전체 event와 이전 history의 compact replay
- next replay: `final_label`별 stable hash 순으로 최대 32행, 총 최대 96행

Spring은 Parquet·manifest·GCS URI를 만들지 않고 재학습용 GCS 권한도 갖지 않는다.
이는 Spring을 학습 artifact schema와 GCS 운영에서 분리해, 온라인 백엔드가 event
선택·PII 제거·HTTP 전송에만 집중하게 하기 위한 의도적인 책임 경계다. ML 입력 형식과
저장 정책이 바뀌면 FastAPI와 ML pipeline만 함께 변경한다.

ML prepare 단계는 training feedback을 canonical row로 정규화하고, RouterBench 원본 전체가
아닌 label별 train 32행 + validation 16행의 deterministic rehearsal set과 합친다. 따라서 한
호출의 기본 rehearsal 상한은 base 144행 + history replay 96행이며 현재 cycle feedback만
전량 유지된다.

## 모델 초기화·승격·보존

- `minor_fine_tune`, `major_consolidation`: 해당 cycle의 `.0` approved anchor에서 weight-only
  warm start한다. optimizer와 scheduler는 새로 만든다.
- `pristine_rebase`: training config의 pinned pristine ModernBERT에서 시작한다.
- RouterArena 평가와 현행 공통 deploy gate를 통과하면 자동 승인되고 `CURRENT.json`이
  원자적으로 이동한다.
- GCS model package는 current, 직전 known-good, 진행 중 cycle anchor만 보호한다. 승인 후
  중복 candidate와 이 경계 밖의 candidate/approved package는 best-effort로 제거한다.
- FastAPI는 `CURRENT.json`을 polling하고 package checksum과 label contract를 확인한 뒤
  load-before-swap 방식으로 반영한다.

GCS의 temporary pipeline artifact는
[`ml/config/gcs_pipeline_artifacts_lifecycle.example.json`](../../ml/config/gcs_pipeline_artifacts_lifecycle.example.json)
같은 lifecycle을 실제 object prefix에 맞게 적용한다. Temporary artifact 전용 bucket은
soft delete의 기본 7일 보존이 비용을 다시 만들 수 있으므로 운영 복구정책과 비용 한도를
확인해 별도로 설정한다. model rollback은 GCS soft delete가 아니라 명시적으로 보호한
known-good approved package가 담당한다.

## GCS 무료 한도 판단

Google Cloud Always Free의 Standard Storage는 월 5 GB-month이며 `us-west1`,
`us-central1`, `us-east1`에만 적용된다. `nam4`, `US` multi-region 또는 캐나다를 포함한
다른 북미 location은 이 조건과 같지 않다. ModernBERT-base safetensors 하나가 약 599MB라
현재 + known-good + anchor의 approved weight만 계산하면 약 1.8GB다.

하지만 이것이 최악의 경우 무료를 보장한다는 뜻은 아니다. pipeline output, package 조립
artifact, 승인 전 candidate, 삭제 후 기본 7일간 과금되는 soft-deleted object가 더해지고
호출 횟수에는 상한이 없기 때문에 5 GB-month를 넘을 수 있다. 따라서 다음을 함께 적용한다.

1. model publish cleanup으로 live package를 current + previous + anchor로 제한한다.
2. pipeline-root에는 1일 TTL lifecycle을 적용한다.
3. 임시 artifact bucket은 durable model bucket과 분리하고, 복구 필요성이 낮은 연구용
   bucket에서는 soft delete 기간을 명시적으로 검토한다.
4. Cloud Billing budget/alert와 GCS storage metric을 별도로 설정한다. budget alert는 과금을
   차단하지 않으므로 실험 횟수에도 운영 상한을 둔다.

## 후속 API 계약으로 구체화한 항목

다음 항목 가운데 version-policy 전환과 canonical digest ledger는 구현했고, gate 분리는
후속 과제로 유지한다.

1. Version trigger는 기본적으로 성공한 HTTP-triggered run 3회다. 개발자 전용 API로
   `successfulHttpRequest`와 `validatedUniqueBatch`를 전환할 수 있다. 후자는 ML
   전처리 후 canonical training row가 1건 이상이고 이전에 commit된 canonical batch
   digest와 다른 run만 count한다. 정책 변경은 다음 major boundary에서 활성화된다. 정확한
   API와 digest/ledger 계약은
   `LLM_Router-serving/docs/api-contract.md` 9장과
   `LLM_Router-serving/docs/retraining-api-implementation-plan.md`를 따른다.
2. minor용 완화 gate와 major/rebase용 강화 gate의 분리. 현재 모든 version은 기존 공통
   RouterArena deploy gate를 통과해야 한다.
3. 호출 수가 아닌 학습효과를 기준으로 하는 semantic version 정책.

개발자 mode 전환 API와 canonical digest 검증은 ML pipeline과 FastAPI에 반영돼 GCP에
배포됐다. 기본 배포 mode는 `successfulHttpRequest`이며 개발자 API는 별도 credential과
feature flag를 설정한 환경에서만 활성화한다.

SQLite 상태를 잃은 뒤 logical version을 복구하는 경우에는 향후 `CURRENT.json`의 logical
version과 batch lineage를 authoritative metadata로 삼는 migration이 필요하다. 현재는 동일
FastAPI state volume이 유지된다는 전제다.
