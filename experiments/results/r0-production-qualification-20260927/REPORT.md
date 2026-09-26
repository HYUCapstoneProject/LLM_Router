# R0 운영 출발 모델 등록 검증 — 2026-09-27

기존 A(RouterBench+xRouteBench train 21,406)의 final checkpoint를 재학습 없이 사용한다. 추론·평가는 8192이며 원래 학습512 기록과 가중치는 보존했다. 원본 archive와 manifest의7개 파일 SHA를 재확인했다.

D_dev 3,581건의 기존 CUDA/fp32/eager/8192 score는 모델·데이터·파일 SHA와 sample ID 및 원본25개 공통열 일치를 확인해 재사용했다. D_dev group bootstrap1,000회·seed42의 pointwise95% 하단을 D_gate 실행 전에 동결했다. 다중비교 보정이나 서비스 품질보장이 아니며, D_dev가 모델 개발에 사용됐다는 한계가 있다.

D_dev conversation8건은 사전 정책(30건 미만)에 따라 초기 등록에서 보고만 하고 품질 미검증으로 표시했다. 통과시키기 위한 양수 하한은 만들지 않았다. 일반 재학습 gate의 분야별 회귀 검사는 유지한다.

D_gate 3,581건을 Mac CPU torch2.5.1/transformers4.48.3 fp32/eager로 새로 평가했고 9개 고정 하한을 모두 통과했다. `gate-metrics.json`과 `initial-qualification.json`이 수치·조건의 원본이다. 평가 시간은418.3초이며 실제 VM 응답시간 측정이 아니다.

초기 등록 통과와 서버 배포·재학습 인수는 구분한다. 실제 VM8192 p50/p95·메모리·동시요청·Spring timeout, 실제202→Vertex→gate→CURRENT→적재, 실패/복구 검증이 남는다. 자동 제출은 false로 유지한다. 기존 운영 VM의 모델과 smoke/empty/V3 설정은 아직 변경하지 않았다.
