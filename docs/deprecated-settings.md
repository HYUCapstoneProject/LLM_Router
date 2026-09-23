# Deprecated settings — 변경 단계와 이전 기록

기준일: 2026-09-23. 두 저장소의 **과거 설정을 찾는 유일한 요약 문서**다. 현재 실행 기준은 [ML 최종 명세](ml/README.md)와 [Serving API 계약](../../LLM_Router-serving/docs/api-contract.md)이다. 이전 문서의 전문은 아래 Git snapshot에서 조회한다. 이력 때문에 날짜별 문서를 다시 만들지 않는다.

`deprecated`라는 파일명은 모든 관련 코드·artifact가 삭제되었다는 뜻이 아니다.

| 상태 | 해석 |
|---|---|
| 대체됨 | 신규 실행의 기준으로 사용하지 않는다. 현재 소유 문서가 우선한다. |
| 운영·재현 호환 보존 | 기존 운영 모델·artifact·fixture 재현에는 유효하다. 새 실험에 자동 적용하지 않는다. |
| 연구 전환 미구현 | 최신 연구 정책은 바뀌었지만 코드가 아직 예전 동작을 가진다. 구현 차이 표에서 추적한다. |
| 과거 관측 | 당시의 실행·배포 기록이다. 현재 cloud 상태나 최신 실험 결과의 증거가 아니다. |

## 1. 초기 기획 → 현재 연구 전제

| 시점·근거 | 이전 설정 | 현재 기준·전환 이유 | 상태 |
|---|---|---|---|
| 2026-07-07 일정 초안 / 07-08 MVP 기준일 | 인프라 우선 일정, 8/31 배포·9/30 결과 동결·11/1 제출, always-Opus 대비 약 3배 절감 목표 | 당시 계획이다. 현행 전제는 캐시·라우팅·feedback의 조건별 비용–품질 검증이며, 절감률과 우승 모델을 미리 보장하지 않는다. 일정은 현재 작업 결정에서 확인한다. | 대체됨 |
| 초기 개요·팀 규칙 | 단일 monorepo 유지 후 3개 저장소로 분리하는 구상, Notion을 기술 결정 SSOT로 설명 | `LLM_Router`와 `LLM_Router-serving`의 책임 경계로 정리했다. 프로젝트 상태·결정은 memorize, 실행 명세는 현재 문서, 결과는 versioned artifact, 협업 추적은 GitHub가 담당한다. | 대체됨 |
| 초기 GitHub 구성 기록 | issue 12개·Project item 수·당시 milestone 상태를 정적 표로 복사 | 유효 협업 규칙과 Project 링크만 [문서 시작점](README.md)에 유지한다. 과거 수치를 현재 진행 상황으로 읽지 않는다. | 대체됨 |

## 2. 초기 인프라 → 관리형 실행과 역할 분리

| 시점·근거 | 이전 설정 | 현재 기준·전환 이유 | 상태 |
|---|---|---|---|
| 2026-08-11 초기안 → 08-14 `353d556` | GKE에서 자체 KFP를 운영하고 Vertex T4를 호출, 일부 문서의 `us-central1` 기본값 | 관리형 Vertex AI Pipelines로 전환했다. KFP SDK 사용과 자체 control plane 운영을 구분한다. 코드 기본 region은 `us-west1`, 실제 자원은 실행 manifest로 고정한다. | GKE·자체 KFP 대체됨 |
| 2026-09-14 `aa7de16` / 09-16 `dbdad2a` | 모든 초기학습·모델 비교도 운영 cloud pipeline에서 실행하는 설명 | 초기학습·비교는 Colab, 운영 ModernBERT feedback 재학습만 Vertex Pipelines/Custom Training을 사용한다. 과거 최초 학습 IR은 회귀·재현 자산으로 남긴다. | 대체됨 |
| 과거 업로드·배포 안내 | GCS source archive, container image, 실행 서버의 역할을 혼용하거나 특정 날짜의 URL·image를 기본값처럼 사용 | GCS는 artifact, Artifact Registry는 image, serving은 실행 환경이다. source archive는 선택적 백업이다. 최신 URI·digest·권한은 해당 실행에서 확인한다. | 설명 대체됨 |

## 3. RouterBench v1/v2 → solve-rate와 통합 연구 계약

| 시점·근거 | 이전 설정 | 현재 기준·전환 이유 | 상태 |
|---|---|---|---|
| 2026-08 v1/v2, `e7011c1`·`ca8d40a` | RouterBench의 최소충분 tier 규칙, low/mid 성공률 0.67 및 high 1/3, unresolved 제외; v2 balanced train 6,256/class와 8,192 token | 기존 v2 checkpoint·base dataset 재현과 운영 호환 계약이다. 신규 연구는 RB+xRB, T1/T2, 공통 초기 weight 1, neural/BGE 512-token 계약이다. 새 실험 설정을 기존 운영 package에 암묵 적용하지 않는다. | **운영·재현 호환 보존** |
| 2026-09-02 `591a47b` | standalone v3 overall solve-rate hard=0.27/easy=0.64 | 기존 v3 config 재현용이다. 통합 연구는 train-only OOF 선택과 별도 label-rule version을 사용한다. A 통합 난이도와 B source-only 실제 세 후보의 threshold 선택도 분리한다. | 운영·재현 호환 보존 |
| 2026-09 최종 연구 정리 | 넓은 backbone sweep, R3를 포함한 4-method 필수 비교, 구현별 다른 sample weighting | 최소 비교는 같은 ModernBERT의 R0/R1/R2와 TF-IDF+LR·frozen BGE+kNN·ModernBERT 표현 비교다. R3와 추가 backbone은 확장이다. 같은 데이터·split·후보·예산을 고정하며 코드 차이는 [구현 차이 표](ml/README.md#implementation-gaps)에서 관리한다. | 연구 전환 미구현 항목 있음 |
| 2026-09 평가 정리 | RouterArena를 epoch마다 확인하거나 승격 gate에 반복 사용; 인지 수준을 후보 모델 성공률과 동일시 | 연구의 반복 gate는 별도 D_gate다. RouterArena는 모델 동결 후 외부 인지 정렬 평가에만 사용한다. 과거 관찰 이력은 공개하며 “처음 보는 평가셋”이라고 쓰지 않는다. 현재 운영 DAG의 RouterArena gate는 아직 별도 전환이 필요하다. | **연구 전환 미구현** |
| 라벨 해석 정정 | RouterArena label을 42개 모델의 성공률 임계값으로 설명하거나 GPT 주석만으로 모델의 인지 기제를 입증 | RouterArena의 Bloom 기반 인지 주석과 RB+xRB의 상대 solve-rate를 구분한다. Blind 주석·사람 검증·통제 문항쌍은 정렬의 증거이며 학습 기제의 인과 증명은 아니다. | 설명 정정 |
| 캐시 평가 정리 | QQP/PAWS 중복 정확도를 실제 답변 재사용 효과로 간주, 반복률·절감 배수 사전 가정 | QQP/PAWS는 진단용이다. C0/C1/C2의 exact/semantic trace, 답변 재사용 검증, 오답·지연·전체 비용을 별도로 측정한다. Reranker는 선택 확장이다. | 대체됨 |

## 4. URI 제출·수동 승격 → inline feedback·자동 정책

| 시점·근거 | 이전 설정 | 현재 기준·전환 이유 | 상태 |
|---|---|---|---|
| 초기 batch 계약 → 2026-09-12 `528ca02` | Spring이 1,000건을 모아 GCS manifest/events를 올리고 URI envelope 제출 | 현행 HTTP는 `feedback-inline-v1`, 전체 `events` 최소 1건이다. FastAPI가 검증·정규화·Parquet/manifest·GCS 게시를 담당한다. 요청의 `eventCount`·`manifestUri`·`eventsUri`는 현행 입력이 아니다. 1,000건은 API 최소 조건이 아니다. | 대체됨 |
| 저장용 feedback-export-v0.1 | 옛 URI batch 예제가 내부 artifact와 같은 schema를 써 현재 HTTP 요청으로 오해할 수 있음 | 내부 manifest/Parquet 계약과 과거 fixture는 유지한다. 같은 버전 이름이 있다고 옛 HTTP envelope를 다시 허용한 것은 아니다. 예제별 용도는 [예제 안내](../../LLM_Router-serving/docs/examples/README.md)에서 구분한다. | **내부·재현 호환 보존** |
| 최초 package 예외 → 2026-09-06 `06a2534` | 초기 baseline·정식 gate 부재로 최초 모델 gate 생략, 수동 qualification/publish 중심 | 평가·gate·자동 승인·CURRENT 게시·serving 적재로 연결한다. 최초 bootstrap 예외를 후속 재학습의 승격 규칙으로 재사용하지 않는다. | 대체됨 |
| 2026-09-12 `a0320ea` | 직전 checkpoint를 계속 이어 학습, 단순 N건 trigger·증분 버전으로 설명 | 성공 3회/cycle, 마지막 승인 `.0` anchor, compact replay와 base rehearsal, 15성공 시 `6.0` pristine rebase 정책이다. optimizer resume와 weight initialization을 구분한다. 세부 정책은 [재학습 장](ml/README.md#retraining)에만 둔다. | 대체됨 |
| Serving `6211b66` → `ba7f202` | 수동 publish·고정 package만으로 운영하는 설명 | 승인 CURRENT watcher와 자동 cycle 정책을 사용한다. 고정 package·local loader는 개발·호환 기능으로 코드에 남아 있다. | 운영·재현 호환 보존 |
| Serving `cb28bf7` → `24e226f` / `eb3bacb` | 상태·current-model·개발자 정책 API가 없다는 gap, inline/status fixture를 앞으로 교체할 목표로만 설명 | endpoint·schema·SQLite 정책·GCS 관측 reader는 존재한다. 옛 rollout 단계 대신 현재 계약을 읽는다. fixture는 실제 배포 응답이나 GPU E2E 완료 로그가 아니다. | 계획 대체됨 |
| 2026-09-23 source 대조 | 상태 조회의 상세 task 추적·10초 TTL·완전 snapshot·heartbeat 단조/10분 stale 검사·publish-result 필수 검증을 완료처럼 서술 | 이 보장은 현재 구현되어 있지 않다. API 소비자가 의존할 수 있는 동작은 [관측 구현 범위](../../LLM_Router-serving/docs/api-contract.md#implementation)에 모았다. | **설명 정정·구현 보완** |
| 2026-09-23 source 대조 | classify confidence를 validation 보정값으로 설명, developer PUT을 2필드 초안으로 설명 | confidence는 raw softmax다. PUT은 mode·activation·expectedRevision·reason 네 필드와 다음 major 경계 예약을 요구한다. 코드·schema를 바꾼 것이 아니라 문서 오류를 고쳤다. | 설명 정정 |
| Serving `b967496`·`b31ff2e` → `6c673f3` | cross-encoder 추가·sentence-transformers 의존 및 점수 parity 실험 설명이 여러 파일에 분산 | 현행 transformers reranker와 고정 image weights를 유지한다. 기본은 비활성이며 cache threshold는 Spring 소유다. backbone 비교가 ModernBERT 전용 자동배포 경로를 확장한 것은 아니다. | 실험 이력 압축·현행 기능 보존 |
| 현행 `.env.example` | 날짜·commit·bucket·v2 path, 빈 baseline·smoke gate 예시를 운영 기본값으로 오해 | 설정 항목·코드 호환 예시는 남겼다. 실제 승인 package·baseline·gate·data/IR 계약을 실행 전에 채운다. 이번 문서 정리가 배포 설정 교체를 의미하지 않는다. | 호환 예시 보존 |

이 절의 `Serving` commit은 LLM_Router-serving 저장소 기준이다. Inline 전환의 serving 근거는 `8fa3672`·`526a820` → 2026-09-12 `24e226f`다. 현재 endpoint와 오류 조건은 API 계약을 우선한다.

## 5. 과거 결과에서 보존할 사실

당시 문서의 수치를 옮긴 요약이며 이번 정리에서 재계산하거나 cloud를 재조회한 결과가 아니다. 현재 최종 데이터·실험의 성능표로 사용하지 않는다.

| 관측 | 보존할 요약 |
|---|---|
| v1 중단 | epoch 1, 3,476 step 중단, final-model 없음. RB validation macro-F1 0.4456, medium F1=0, RouterArena 0.3011 |
| v2 balanced | train 18,768 / validation·test 각 3,218. T4 FP16·8,192 token·3 epoch·7,038 step·약 5시간48분. RB val epoch2 0.4505 → epoch3 0.4352, RouterArena 0.3293 → 0.3212 |
| TF-IDF 진단 | 같은 v2 validation macro-F1 0.4198. 모델 간격은 관측이며 label ceiling이나 source shortcut의 인과 증거가 아니다. |
| 2026-08-16 최초 package | `router-modernbert-base-routerbench-20260814-45dcc6d`, v2 epoch3. 당시 baseline·정식 규칙 부재로 최초 gate 생략 |
| 재현 공백 | epoch2 weight는 최근 checkpoint 2개 retention으로 삭제됐다. 학습 당시 일부 코드가 untracked여서 `45dcc6d` SHA만으로 전체 재현할 수 없다. |
| 배포·smoke 기록 | 08-16 GCE 호출, 08-17 compiler 추가 후 GPU smoke, 09-12~16 API/version/모델 동기화 기록. 최신 유효 feedback GPU E2E·현재 live health 증명과 구분한다. |

## 6. 문서 통합과 원문 조회

| 저장소·snapshot | 보존한 범위 |
|---|---|
| LLM_Router `fa09cb8931526faeb3ed44f934b032adbb5bcce3` | 첫 통합 전 ML 문서·리포트 22개 |
| LLM_Router `2ff4064daca9054a406645debdd7d700c102a9cd` | 2026-09-20의 ML 7문서 및 공통 개요·팀 규칙·GitHub 설정·일정·MVP, 이번 재정리 전 상태 |
| LLM_Router-serving `1ad9c89` | 이번 재정리 전 실행·API·아키텍처·handoff·관측·feedback·예제 등 Markdown 13개 |
| 2026-09-23 통합 | LLM_Router의 `docs` 13개 → 현재 안내·ML 명세·이력 3개. Serving Markdown 13개 → 실행·API·예제 안내 3개 |

원본은 각 저장소 루트에서 조회한다. 현재 작업 파일을 되돌릴 필요 없이 `git show`로 읽거나 별도 폴더에 추출한다.

```bash
# LLM_Router: 기존 문서 목록과 특정 파일 조회
git ls-tree -r --name-only 2ff4064 -- docs
git show '2ff4064:docs/team rules.md'
git show 2ff4064:docs/ml/execution-and-artifacts.md

# LLM_Router: 첫 통합 전 ML 원문을 별도 보관
git archive --format=zip --output=/tmp/llm-router-ml-before-consolidation.zip fa09cb8 docs/ml

# LLM_Router-serving: 이전 API/계획/예제 조회
git ls-tree -r --name-only 1ad9c89 -- README.md docs
git show 1ad9c89:docs/retraining-observability-plan.md
```

**Git에 원본 데이터·대형 로그·가중치가 전부 보관된 것은 아니다.** `ml/data/`, `ml/artifacts/`는 ignore 대상이다. Drive/GCS/로컬의 보관 위치와 checksum을 따로 확인한다. 문서 삭제와 함께 코드·config·JSON fixture·원본 artifact를 삭제하지 않았다. 원본 artifact가 없으면 문서 수치만으로 재현했다고 표시하지 않는다.

이후 변경은 현재 소유 문서를 수정하고, 이 파일에 대체된 설정·이유·상태·근거 commit을 짧게 남긴다. 과거 설정 전문과 날짜별 문서 사본은 추가하지 않는다.
