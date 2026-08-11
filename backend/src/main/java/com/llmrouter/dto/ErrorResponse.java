package com.llmrouter.dto;

import lombok.Builder;
import lombok.Getter;

/**
 * 04 API Contract - Error Format Draft 를 따른다.
 *
 * requestId 는 서버 로그의 requestId 와 같은 값이다.
 * 사용자가 에러를 제보하면 이 값으로 로그를 바로 찾을 수 있다.
 */
@Getter
@Builder
public class ErrorResponse {

    /** 사용자에게 보여줄 메시지 */
    private String error;

    /** 클라이언트가 분기에 쓸 안정적인 코드 */
    private String code;

    private String requestId;
}
