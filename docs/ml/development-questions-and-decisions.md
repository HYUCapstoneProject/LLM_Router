# ML 개발 질문·고민·의사결정 기록

상태: **설계 근거 기록**

기준일: 2026-08-10

## 1. 문서 목적

이 문서는 ML 부분을 개발하면서 제기한 질문, 확인한 사실, 선택한 방향과 아직 남은 고민을
한 곳에 기록한다. 단순한 작업 일지가 아니라 “왜 현재 구조가 되었는가”를 나중에 설명하기
위한 의사결정 지도다.

- 현행 작업 상태와 체크박스: `docs/ml/workstream-checklist.md`
- 팀에서 확정할 계약: `docs/ml/backend-infrastructure-agreement.md`
- 세부 알고리즘과 artifact 계약: RouterBench, RouterArena, ModernBERT 각 명세

상태 표기는 `결정`, `확인`, `실험 필요`, `팀 협의`로 통일한다.

## 2. RouterBench 데이터와 난이도 라벨

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| RouterBench의 규모와 형식은 무엇인가 | 공식 Pickle의 canonical 질문 36,497개 중 초기 핵심 범위 34,778행을 라벨링했다. 감사·학습용 Parquet/JSON으로 변환한다. | 확인 |
| 난이도를 무엇으로 정의할 것인가 | 교과 난이도가 아니라 질문을 만족스럽게 해결할 수 있는 최소 모델 capability tier로 정의한다. | 결정 |
| 대표 모델 3개만 사용하면 충분한가 | 특정 모델의 우연한 강·약점에 민감하다. 11개 모델을 empirical low/medium/high tier로 나누고 tier consensus를 사용하면 모델군 수준의 해결 가능성을 반영해 일반화에 유리하다. | 결정 |
| solve rate는 주 라벨 기준인가 | 전체 solve rate와 cheapest-correct model은 유용한 진단값이지만 최종 3-class label은 tier별 consensus와 threshold로 정한다. | 결정 |
| 라벨링 예시는 어떻게 해석하는가 | low tier가 안정적으로 성공하면 easy, low는 부족하고 medium이 성공하면 medium, high가 필요하거나 전체가 실패하면 hard다. | 결정 |
| 100행 smoke로 라벨 품질을 확정할 수 있는가 | smoke는 schema, 다운로드, 라벨링, 저장 경로의 동작 확인용이다. threshold 품질 판단은 전체 분포와 실제 Claude 표본 검증이 필요하다. | 확인 |
| RouterBench 원본을 그대로 학습에 넣어도 되는가 | few-shot wrapper에서 실제 query를 분리하고, 중복 그룹으로 split 누수를 막고, 정답·모델 성능 컬럼을 학습 입력에서 제거해야 한다. | 결정 |
| 전체 데이터 처리 결과는 무엇인가 | 핵심 34,778행 라벨은 easy 12,765, medium 5,348, hard 16,665이고 학습 projection은 34,764행이다. | 확인 |
| RouterBench 모델이 Claude가 아닌데 라벨을 전이해도 되는가 | capability proxy로 사용할 수 있으나 실제 Claude tier의 충분성을 직접 증명하지 않는다. 소규모 Claude 표본 검증이 필요하다. | 실험 필요 |
| continuous threshold와 RAG 관련 평가를 유지할 것인가 | 3-class baseline 이후 sensitivity 분석 후보로 남아 있다. 서비스 RAG는 구현 범위에서 제외했다. | 실험 필요 |

## 3. RouterArena 평가를 설계하며 해결한 문제

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| RouterArena는 질문별 Bloom 6단계인가 | README는 Bloom 6단계를 설명하지만 공개 Parquet의 질문별 `Difficulty`는 이미 easy/medium/hard 3단계다. | 확인 |
| Bloom 6단계를 2·2·2로 합쳐야 하는가 | 공개 데이터에 질문별 6단계 값이 없으므로 변환하거나 역추정하지 않는다. | 결정 |
| RouterBench와 RouterArena의 같은 label을 정답처럼 비교해도 되는가 | 생성 기준이 다르다. accuracy/F1은 외부 alignment proxy이며 운영 Claude 품질의 직접 증거가 아니다. | 결정 |
| 무엇을 주 분석으로 볼 것인가 | ModernBERT 예측 easy/medium/hard 세 그룹을 만든 뒤 각 그룹 내부 RouterArena label 분포 `P(RouterArena label | predicted group)`를 시각화한다. | 결정 |
| 학습 데이터와 평가 데이터가 겹치면 어떻게 하는가 | 정규화된 질문 hash로 RouterBench 전체 학습 artifact와 exact overlap을 제거한다. 현재 전체 기준 22행이 제외됐다. | 결정 |
| robustness 데이터는 라벨이 없는데 어떻게 평가하는가 | `Global Index`로 full split의 공식 label과 원 질문을 결합해 원본/변형 예측 일관성을 본다. | 결정 |
| 전처리와 평가가 같은 자원을 써야 하는가 | 입력 조립·중복·누수 검사는 CPU, checkpoint 추론은 full 반복 평가에서 GPU를 기본으로 한다. 작은 smoke는 CPU로 가능하다. | 결정 |
| GPU 평가가 낭비인가 | 학습보다 가볍지만 전체 8천여 행, 긴 입력과 반복 checkpoint 비교에서는 GPU가 시간을 줄인다. 비용 최적화가 필요하면 CPU smoke와 GPU full 평가를 분리한다. | 결정 |
| 어떤 평가가 아직 남았는가 | 전체 checkpoint 평가, RouterBench holdout evaluator와 robustness consistency가 아직 필요하다. | 실험 필요 |

## 4. ModernBERT 학습과 checkpoint

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| checkpoint 경로는 무엇이며 왜 필요한가 | tokenizer, model weight, config와 label mapping을 저장한 모델 디렉터리다. 평가·재로딩·서빙이 같은 모델 버전을 재현하려면 필요하다. | 확인 |
| `id2label` 순서는 어디서 확인하는가 | checkpoint의 `config.json`과 학습 config/metrics에서 확인한다. 계약은 `0=easy, 1=medium, 2=hard`다. | 결정 |
| 로컬 smoke 코드와 Kubeflow 학습 코드는 달라야 하는가 | 공통 학습 코어와 동일 CLI를 사용하고 경로, device, precision, batch와 resource만 바꾼다. 환경별 구현 drift를 막는다. | 결정 |
| 인프라가 없을 때 GPU smoke를 어떻게 하는가 | VS Code Colab 확장과 업로드 archive로 100행 CUDA smoke를 수행했다. Colab 파일시스템은 임시이므로 checkpoint 경로와 출력 존재를 확인해야 한다. | 확인 |
| 작은 smoke metric을 성능으로 해석해도 되는가 | optimizer step 2 결과의 accuracy 0.3, macro-F1 약 0.288은 학습 성공 판단용이 아니다. CUDA 실행, 저장과 reload 통과가 핵심 결과다. | 확인 |
| 로컬·Colab·Docker runtime을 어떻게 맞추는가 | PyTorch/CUDA/Transformers 등 runtime 계약을 versioned JSON과 metrics에 기록하고 Docker/GKE 기준 버전을 고정한다. | 결정 |
| T4 전체 학습 설정은 확정인가 | max 8192, FP16, batch 1, accumulation 8, gradient checkpointing은 baseline 시작값이다. 전체 학습 후 early stopping과 checkpoint 선택 정책을 보완한다. | 실험 필요 |
| Spot GPU가 중단되면 어떻게 하는가 | retry만으로 긴 학습을 보호하기 어렵다. periodic checkpoint와 resume 정책이 아직 필요하다. | 실험 필요 |

## 5. 로컬, Docker, Pod와 Kubeflow 경계

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| 초기 라벨링·전처리도 서버에서 해야 하는가 | 일회성 RouterBench 처리는 로컬 CPU가 비용과 설정 부담이 작다. 검증된 `training.parquet`만 GCS에 올린다. | 결정 |
| feedback 처리도 로컬에서 하면 되는가 | 반복 자동화, lineage와 재현성이 필요하므로 feedback prepare/merge는 Kubeflow CPU Pod에서 수행한다. | 결정 |
| 같은 CPU image에 로컬 초기 처리와 서버 feedback 처리가 같이 있어도 되는가 | image는 실행 가능한 기능 묶음이고 component가 어떤 CLI를 실행할지 정한다. 같은 image를 재사용해도 Pod와 artifact 경계는 분리된다. | 결정 |
| CPU 작업도 image가 필요한가 | 로컬 초기 처리에는 필수가 아니지만 GKE의 feedback component는 재현 가능한 CPU image가 필요하다. | 확인 |
| Pod이 무엇인가 | Kubernetes가 container를 실행·스케줄링하는 최소 배포 단위다. Kubeflow component 한 단계가 보통 하나의 CPU 또는 GPU Pod로 실행된다. | 확인 |
| Kubeflow에 라벨링 전용 기능이 있는가 | Kubeflow는 도메인 라벨 알고리즘을 제공하지 않고 실행, artifact, cache, retry와 lineage를 관리한다. 프로젝트 전처리기는 그대로 재사용한다. | 확인 |
| CPU와 GPU 기능을 한 image로 합칠 것인가 | dependency와 크기, scheduling을 분리하기 위해 data CPU, training GPU, evaluation GPU, promotion CPU 네 image로 나눴다. | 결정 |
| Docker image와 component는 같은 것인가 | image는 실행 환경, component는 image·명령·입출력·자원 계약, pipeline은 component 실행 순서다. | 확인 |

## 6. GCS, GKE와 cloud 준비

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| GCS 안에 GKE가 있는가 | 아니다. GCS는 artifact 저장소, Artifact Registry는 image 저장소, GKE는 image를 Pod로 실행하는 Kubernetes cluster다. | 확인 |
| CPU/GPU image는 어디에 들어가는가 | Artifact Registry에 push하고 GKE Pod가 pull한다. 데이터·checkpoint·metric은 GCS에 저장한다. | 확인 |
| 라벨링·전처리를 로컬에서 끝내고 결과만 올려도 되는가 | 초기 학습은 그렇게 한다. feedback 재학습 때만 CPU/GPU component를 실행하면 자원을 절약할 수 있다. | 결정 |
| 서버가 없거나 schema가 미정일 때 무엇을 할 수 있는가 | pure core, CLI, fixture, contract validator, Docker smoke, KFP IR compile, 문서와 테스트를 준비할 수 있다. 실제 IAM·GCS 전달·Pod scheduling은 cluster가 필요하다. | 확인 |
| “cloud smoke 차단”은 작은 데이터 제한을 푸는 뜻인가 | 아니다. 로컬 구현은 끝났지만 API, bucket, registry, cluster, IAM이 없어 실제 cloud 실행을 못 한다는 뜻이었다. | 확인 |
| 현재 cloud는 준비됐는가 | project 접근만 가능하고 확인 당시 GCS bucket 0개, GKE·Artifact Registry API 비활성, kubectl context 없음이었다. 생성 전 이름·region·예산 합의가 필요하다. | 팀 협의 |
| 서버 확보 전 할 수 있는 일을 모두 했는가 | 코드·image smoke·KFP compile 기반은 완료했지만 full evaluator, robustness, resume, gate threshold와 계약 확정은 남아 있다. | 실험 필요 |

## 7. Feedback 재학습과 역할 분담

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| feedback 누적 재학습 코드가 준비됐는가 | 합성 schema 기준 validator, normalize, dedup, RouterBench merge와 CPU→GPU 재학습 pipeline은 구현됐다. 실제 DB export 계약과 cloud 검증은 남았다. | 확인 |
| ML 담당자가 온라인 DB와 API까지 구현해야 하는가 | 아니다. ML은 batch 계약, adapter/validator, 학습·평가·package/gate를 담당하고 온라인 팀이 수집·저장·export·serving을 담당한다. | 결정 |
| feedback schema가 확정되기 전에 image를 만들 수 있는가 | versioned adapter와 합성 fixture로 image를 만들 수 있다. 실제 schema가 바뀌면 adapter와 fixture를 함께 갱신한다. | 결정 |
| 모든 만족 질문을 즉시 학습 라벨로 사용해도 되는가 | 사용자 선호, 일시적 응답 오류와 중복·충돌이 섞이므로 곧바로 채택하면 안 된다. 품질 필터와 반복 근거가 필요하다. | 결정 |
| ML이 escalation 여부를 다시 검사해야 하는가 | upstream export가 유효 feedback만 포함함을 계약으로 보증한다면 비즈니스 규칙을 중복 판단할 필요는 없다. 다만 provenance/audit 필드는 남기고 schema validator는 수행한다. | 팀 협의 |
| 재학습 N은 어떻게 정하는가 | dedup 후 accepted count, class별 최소량, 최대 대기 기간, GPU 비용과 성능 변화를 함께 봐야 한다. 현재 숫자는 미정이다. | 팀 협의 |
| 기존 holdout과 feedback이 겹치면 어떻게 하는가 | validation/test overlap은 제외하고 train과 충돌하면 versioned 정책으로 교체·집계한다. | 결정 |

## 8. 평가, package와 배포 게이트

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| Kubeflow component는 CPU와 GPU 두 개뿐인가 | 자원 종류는 CPU/GPU지만 책임 경계를 위해 현재 component는 feedback prepare, merge, train, evaluate, package, gate의 6개다. | 확인 |
| RouterArena 예측 그룹은 무엇인가 | ModernBERT가 easy/medium/hard로 예측한 세 집단이다. 각 집단 안의 공식 RouterArena label 분포로 모델의 정렬 특성을 본다. | 확인 |
| model package manifest는 무엇인가 | checkpoint 파일 목록, checksum, label order, model/data/code/evaluation version과 metric 위치를 기록한 인수인계 명세다. | 확인 |
| package validator는 왜 필요한가 | 파일 누락, checksum 손상, 잘못된 label 순서와 평가 증거 누락을 serving 전에 차단한다. | 확인 |
| deploy gate가 바로 GKE에 배포하는가 | 아니다. metric 규칙을 통과한 candidate를 만들고 실패 task를 차단한다. 실제 approved 승격과 serving 전환은 별도 계약이다. | 결정 |
| Kubeflow의 배포 역할은 어디까지인가 | 학습→평가→package→gate와 evidence/lineage 생성까지다. online health check, process 전환과 rollback은 운영 영역이다. | 결정 |
| gate 기준은 확정됐는가 | core는 구현됐지만 실제 baseline과 허용 하락폭은 full 학습·평가 후 정해야 한다. | 실험 필요 |

## 9. 범위에서 제거하거나 다시 정의한 요소

| 질문·고민 | 얻은 답과 현재 판단 | 상태 |
| --- | --- | --- |
| RAG가 무엇이며 이 프로젝트에서 필요한가 | 외부 지식을 검색해 생성 입력에 붙이는 구조다. 구현하지 않으며 유사 질문 재사용은 semantic cache가 일부 다른 목적을 수행한다. | 결정 |
| QQP threshold는 ML이 정해야 하는가 | QQP는 캐시된 질문과 새 질문의 의미 동일성 임계값 평가다. 온라인 cache 팀 책임이고 ML DAG에서 제거했다. | 결정 |
| 초기 개요의 Haiku/Sonnet/Opus mapping은 확정인가 | 아니다. Claude 제품 변화에 대응하도록 backend routing config에서 협의·관리해야 한다. | 팀 협의 |
| 초기 개요의 “Kubeflow가 라벨링부터 배포” 표현은 정확한가 | 현재는 초기 라벨링을 로컬에서 하고 Kubeflow는 학습 이후 및 feedback 루프를 담당한다. gate 이후 실제 서빙 배포는 아직 협의 대상이다. | 결정 |
| MVP 문서와 최종 ML 구조가 다른 것은 충돌인가 | MVP 문서는 역사적 최소 범위다. 현행 구조와 상태는 총괄 체크리스트를 기준으로 하고 세부 명세는 근거 문서로 사용한다. | 결정 |

## 10. 현재 남아 있는 핵심 질문

다음 항목은 코드 기본값과 팀의 운영 결정을 구분해야 한다.

1. 실제 Claude model ID, label mapping, timeout과 fallback
2. feedback export가 보증하는 이벤트 조건과 ML validator의 최소 책임
3. 유효 feedback trigger N, class별 최소량, 최대 대기 기간과 실행 주체
4. full RouterBench/RouterArena baseline과 deploy gate threshold
5. candidate 승인 주체, approved GCS 경로, FastAPI 적용·health check·rollback 방식
6. GCP project·region·bucket·registry·cluster·namespace·IAM·예산
7. RouterArena GPU full 평가와 CPU smoke를 운영에서 언제 사용할지
8. Spot 중단을 견딜 checkpoint resume와 전체 학습 완료 기준

이 질문들의 합의 결과는 `docs/ml/backend-infrastructure-agreement.md`에 기록하고, 실제
진행 상태는 총괄 체크리스트에서 갱신한다.
