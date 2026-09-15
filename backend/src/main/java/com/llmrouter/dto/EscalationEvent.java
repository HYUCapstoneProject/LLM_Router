package com.llmrouter.dto;

import java.time.Instant;

/**
 * FastAPI {@code POST /escalate} 가 받는 event 하나 (04 API Contract).
 *
 * <p><b>정확히 12필드다.</b> 계약은 요청 JSON 의 알 수 없는 필드를 거부하므로 여기에
 * 필드를 하나라도 더하면 배치 전체가 422 로 떨어진다. 특히 {@code eventCount},
 * {@code manifestUri}, {@code eventsUri} 는 <b>보내면 안 된다</b> — FastAPI 가 내부에서
 * 만든다. 게이트웨이에 GCS SDK 를 들이지 않기 위한 경계다.
 *
 * <p>시각은 {@link Instant} 다. timezone 없는 {@code LocalDateTime}, epoch 숫자,
 * 문자열 boolean 은 쓰지 않는다.
 *
 * @param eventId            고유 escalation ID. {@code esc-{escalation_log.id}}
 * @param requestId          최초 사용자 요청 추적 ID
 * @param question           PII 제거가 끝난 질문 (아래 주의 참고)
 * @param initialLabel       라우터의 최초 판정
 * @param initialModel       최초 응답 모델. {@code escalation_path[0]}
 * @param finalLabel         사용자가 만족한 티어 = 정정 라벨
 * @param finalModel         만족 응답 모델. {@code escalation_path[last]}
 * @param userSatisfied      반드시 true. 아래 주의 참고
 * @param escalationCount    최초 응답 이후 상향 횟수, 1 이상
 * @param createdAt          event 생성 시각 (UTC)
 * @param routerModelVersion 최초 classify 응답의 modelVersion
 * @param piiRedacted        아래 주의 참고
 *
 * <h2>⚠️ 두 boolean 의 실제 의미</h2>
 *
 * <p><b>{@code userSatisfied} 는 추정치다.</b> 계약은 "최종 상향 응답에 사용자가
 * 만족했다"를 요구하지만, 우리 UI 에는 명시적 만족 버튼이 없다. 실제로 저장되는
 * {@code escalation_log.user_satisfied} 는 "체인이 여기서 끝났다"는 뜻이라,
 * 사용자가 그냥 창을 닫은 경우와 구분되지 않는다. 더 상향하지 않았다는 것을
 * 만족으로 간주한 값이며, 이 가정은 실험 결과 해석에 그대로 남는다.
 *
 * <p><b>{@code piiRedacted} 는 상수 true 이고 redaction 을 수행한 결과가 아니다.</b>
 * 이 서비스는 실사용자가 없는 실험용이고 개인정보가 포함된 질문을 넣지 않는다는
 * 전제로 운영한다. 따라서 이 값은 "제거했다"가 아니라 <b>"PII 가 유입되지 않는
 * 경로다"라는 선언</b>으로 읽어야 한다. 실사용자 트래픽을 받게 되면
 * {@code FeedbackBatchService} 에 실제 탐지·마스킹을 넣기 전까지 이 배치를 보내면 안 된다.
 */
public record EscalationEvent(
        String eventId,
        String requestId,
        String question,
        RouterTier initialLabel,
        String initialModel,
        RouterTier finalLabel,
        String finalModel,
        boolean userSatisfied,
        int escalationCount,
        Instant createdAt,
        String routerModelVersion,
        boolean piiRedacted
) {
}
