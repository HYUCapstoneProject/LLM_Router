# 라우터 모델 승격 기록

> 현재 운영 정책(2026-09-06): 아래 최초 승격 기록은 당시의 수동 절차를 보존한 이력이다.
> 이후 pipeline은 deploy gate를 통과한 candidate를 자동 승인하고 `CURRENT.json`을
> 갱신하며, FastAPI는 해당 pointer를 polling해 검증된 model을 자동 반영한다. 사람의
> 사전 승인 단계는 없지만 deploy gate, package checksum 검증, 이전 model 유지 및
> 명시적 rollback 수단은 유지한다.

상태: **운영 기록 문서**

이 문서는 `router-modernbert` 모델 패키지가 candidate에서 approved로 올라간 이력을
시간순으로 남긴다. 각 항목은 "어떤 가중치가, 어떤 근거로, 누구의 승인으로 서빙에
들어갔는가"에 답할 수 있어야 한다.

승격 절차 자체의 계약은
[`gcp-rollout-and-serving-integration-plan.md`](gcp-rollout-and-serving-integration-plan.md)
G-8·G-9와 `LLM_Router-serving/docs/serving-and-retraining-architecture.md` 6장을 따른다.

## 기록 형식

| 항목 | 설명 |
| --- | --- |
| model_version | 패키지 manifest의 `model_version` |
| 출처 | 어떤 학습 run의 어떤 checkpoint인가 |
| gate | deploy gate 통과 여부. 생략했다면 그 사유 |
| 승인자 / 승인일 | `approved_by` / `approved_at` |
| 비고 | 정식 경로에서 벗어난 부분, 알려진 한계 |

---

## 1. `router-modernbert-base-routerbench-20260814-45dcc6d`

기준일: 2026-08-16

상태: **approved 완료 (GCS 업로드 대기)**

| 항목 | 값 |
| --- | --- |
| `approved_by` | `myday2day@hanyang.ac.kr` |
| `approved_at` | 2026-08-16T11:45:03Z |
| `manifest_sha256` | `15198d6c4c16d33891fdd9e49a4b4143779d4d58c97c9ceb8155d80c495026c8` |
| `previous_model_version` | `null` (최초 승인 — `CURRENT.json` 이때 생성) |
| `pipeline_run_id` | `null` (파이프라인 산출물이 아님) |

승격 트리는 GCS 레이아웃 그대로 **이 저장소**에 있다
(`ml/artifacts/model-registry/`). gcloud 토큰 만료와 bucket 미생성(G-2) 때문에
실제 GCS 업로드만 남았다.

```bash
gcloud storage rsync --recursive \
  ml/artifacts/model-registry/llm-router-mlserver gs://llm-router-mlserver
```

**레지스트리는 ML 쪽 자산이다.** candidate/approved/`CURRENT.json`과 publish·approve
도구는 전부 `LLM_Router`가 소유하며 `LLM_Router-serving`에는 두지 않는다. FastAPI
서버는 승인된 패키지를 GCS에서 **내려받아 쓰기만** 한다. 서버가 보유하는 것은
`router_model_cache_dir`(`var/models/<version>`)에 내려받은 사본 하나뿐이다.

승인 후 서빙 검증 결과:

| 확인 | 결과 |
| --- | --- |
| `verify_package(require_approved=True)` on `approved/` | 적재 허용 |
| 같은 검사 on `candidate/` | 거부 (`gate_status='candidate'`) |
| FastAPI 기동 → `GET /health` | 200, 두 모델 loaded |
| `POST /router/classify` | 200, `modelVersion`이 승인 버전으로 반환 |

### 출처

| 항목 | 값 |
| --- | --- |
| 학습 run | `modernbert-router-balanced-v2` |
| checkpoint | `final-model` = epoch 3 / optimizer step 7,038 |
| 원본 위치 | `MyDrive/LLM_Router/training/modernbert-router-balanced-v2/final-model` |
| 무결성 | 7개 파일 SHA-256이 `artifact_manifest.json`과 전부 일치 (2026-08-16 확인) |
| 라벨 순서 | `easy=0, medium=1, hard=2` — 서빙 계약과 일치 |

### epoch 3을 쓴 이유

[v2 학습 리포트](training-reports/modernbert-v2-balanced-training-report.md) 13장은
**epoch 2 checkpoint로 교체**할 것을 1순위로 권고한다. macro F1이 0.4505(epoch 2) 대
0.4352(epoch 3)로 epoch 2가 낫고, RouterBench와 RouterArena 양쪽에서 교차 확인됐다.

그럼에도 epoch 3을 쓴 것은 **epoch 2 가중치가 존재하지 않기 때문**이다. 학습이
`keep_last_checkpoints=2`로 실행되어 마지막 2개(`step-00007000`, `step-00007038`)만
남기고 epoch 2의 `step-00004692`를 삭제했다. 2026-08-16 Drive 전체를 검색해 확인한
결과 남아 있는 checkpoint는 v1 2개 + v2 2개뿐이며 epoch 2는 어디에도 없다.
`routerarena-epoch-evaluations/epoch-02/`와 `observation-06`에는 지표와 예측만 있고
가중치가 없다.

epoch 2를 되살리려면 seed 42로 step 4,692까지 재학습해야 하며 T4 기준 약 3시간
52분이 든다(원 학습 7,038 step에 5시간 48분). 서빙 통합을 먼저 뚫는 편이 낫다고
판단해 epoch 3으로 진행했다. 모델 교체는 환경변수 2개 + 재시작이므로 나중에
epoch 2를 확보하면 그때 갈아끼운다.

### 패키징

| 항목 | 값 |
| --- | --- |
| schema | `model-package-v1` |
| 조립 명령 | `modernbert-package assemble` |
| metadata | `ml/config/package_metadata.json` (이 run과 일치) |
| 동봉 평가 | RouterArena `epoch-03/metrics.json` → `routerarena_metrics.json` |
| 검증 | `modernbert-package validate` → `valid: true` |
| candidate 경로 | `gs://llm-router-mlserver/models/router-modernbert/router-modernbert-base-routerbench-20260814-45dcc6d/candidate/` |

조립·게시·승인은 전부 `LLM_Router`(ML) 쪽에서 수행했다. `LLM_Router-serving`은
이 과정에 관여하지 않으며 결과물만 소비한다.

`model_version`의 날짜 `20260814`는 학습 완료 시각 2026-08-14T19:36:02Z(UTC) 기준이다.

### gate: 생략 (약식 승격)

deploy gate를 거치지 않았다. baseline metrics와 versioned `deploy_gate_rules.json`이
아직 없어서(G-8 미완) 정식 gate를 실행할 수 없기 때문이다. 따라서 이 패키지는
**"기준 대비 성능이 검증된 모델"이 아니라 "형식이 검증된 첫 서빙 모델"**이다.

이 모델이 최초 approved가 되면 이후 재학습 candidate의 비교 기준(baseline)이 되므로,
G-8에서 baseline metrics를 고정할 때 이 모델의 RouterArena 지표를 사용한다.

### 승인

2026-08-16 `myday2day@hanyang.ac.kr`이 아래 명령으로 승인했다.

```bash
modernbert-publish approve \
  --root-uri gs://llm-router-mlserver/models/router-modernbert \
  --model-version router-modernbert-base-routerbench-20260814-45dcc6d \
  --approved-by "myday2day@hanyang.ac.kr"
```

이 명령은 패키지 재다운로드 → 재검증 → `gate_status: approved`와
`approved_at`/`approved_by` 스탬프 → `approved/` 업로드 → `CURRENT.json` 포인터 생성을
수행한다. 최초 승인이므로 `CURRENT.json`이 이때 처음 만들어졌고, 이후 rollback의
기준점이 된다.

당시 정책에서 approve는 `gate_status`를 다시 쓰는 유일한 단계였고 자동화 대상이
아니었다. 승인자 식별자를 패키지에 남기며 사람이 직접 판단해 실행했다. 현재는 같은
필드에 자동화 주체 `vertex-pipeline-deploy-gate`를 기록한다.

### 알려진 한계

1. **`code_commit: 45dcc6d`의 provenance가 부정확하다.** 이 커밋에는 학습 진입점
   `modernbert_colab_training.py`, `modernbert_checkpointing.py`,
   `ml/config/modernbert_t4_balanced_v2.json`이 존재하지 않고(전부 untracked),
   `modernbert_training.py`도 +611/−62줄 다르다. 즉 이 커밋을 체크아웃해도 이 학습을
   재현할 수 없다. 하이퍼파라미터 값 자체는 패키지 안 `training_config.json`에,
   학습 데이터는 `training_data_sha256: 15a485c0…`로 보존되어 있으므로 실제 공백은
   학습 루프·드라이버 코드다. 다음 학습부터는 실행 전 커밋하고 그 SHA를 기록한다.
2. **gate 미통과.** 위 참조.
3. **성능이 v2 리포트의 최선이 아니다.** epoch 2 대비 macro F1 −0.0153.
