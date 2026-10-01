package com.llmrouter.client;

/**
 * 질문이 서비스 입력 한도(BGE 토큰 512)를 넘어 FastAPI 가 거절했다.
 *
 * 장애가 아니라 사용자 입력 문제라 RouterClientException 을 상속하지 않는다.
 * 상속하면 ChatService 가 캐시 우회·MEDIUM 폴백으로 삼켜 버리고,
 * 캐시는 못 쓰고 분류만 되는 반쪽 처리가 된다.
 */
public class MessageTooLongException extends RuntimeException {

    public MessageTooLongException(String message, Throwable cause) {
        super(message, cause);
    }
}
