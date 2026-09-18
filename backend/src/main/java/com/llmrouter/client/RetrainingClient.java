package com.llmrouter.client;

import com.llmrouter.dto.EscalateAcceptance;
import com.llmrouter.dto.FeedbackBatchRequest;
import com.llmrouter.dto.RetrainingStatus;
import com.llmrouter.dto.RouterModelStatus;

/**
 * FastAPI 의 재학습 계약 (04 API Contract).
 *
 * <p>{@link RouterClient} 와 분리한 이유는 호출 성격이 완전히 다르기 때문이다.
 * 그쪽은 <b>사용자 요청마다</b> 도는 추론이라 3초 타임아웃에 실패하면 폴백하면 그만이다.
 * 이쪽은 <b>배치 단위</b>로 어쩌다 한 번 돌고, 실패를 삼키면 안 되며, 무엇보다
 * <b>성공한 배치 하나가 팀원 쪽 모델 버전을 한 칸 올린다.</b> 같은 인터페이스에 두면
 * 추론 클라이언트를 stub 으로 바꿀 때 재학습 제출까지 딸려 stub 이 된다.
 *
 * <p>여기에 GCS SDK, Vertex SDK, bucket 정보는 들어가지 않는다. FastAPI 가 artifact 를
 * 만들고 pipeline 을 제출한다. 게이트웨이는 event 를 JSON 으로 넘기고 상태를 조회할 뿐이다.
 */
public interface RetrainingClient {

    /**
     * POST /escalate — 피드백 배치를 제출한다.
     *
     * <p><b>성공하면 모델 버전이 올라간다.</b> HTTP 호출 1회가 version trigger 1회다.
     * 같은 {@code batchId} + 같은 event 내용은 멱등이라 기존 pipelineRunId 로 202 가
     * 다시 오지만, batchId 를 바꾸면 새 배치이므로 버전이 또 올라간다.
     * 개발·테스트에서 batchId 를 매번 새로 만들지 않는다.
     *
     * @throws RetrainingClientException 409/422 등 재시도하면 안 되는 오류 포함.
     *                                   {@link RetrainingClientException#code()} 로 분기한다.
     */
    EscalateAcceptance submit(FeedbackBatchRequest request);

    /**
     * GET /retraining-runs/{batchId} — 학습 단계와 활성화 상태를 조회한다.
     *
     * <p>실패·거부·취소도 HTTP 200 으로 온다. 200 을 성공으로 읽지 않는다.
     */
    RetrainingStatus status(String batchId);

    /** GET /router/model/current — 승인 모델과 실제 serving 모델을 조회한다. */
    RouterModelStatus currentModel();
}
