package com.llmrouter.service;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * exact match 캐시용 질문 해시.
 *
 * <p>정규화를 Java 한 곳에서만 한다. SQL 에서도 같은 정규화를 하면 두 구현이
 * 어긋나는 순간 인덱스 조회가 예외 없이 조용히 빗나간다. DB 는 저장된 값을
 * 그대로 비교만 한다.
 *
 * <p><b>대소문자는 건드리지 않는다.</b> exact match 의 유일한 존재 이유가
 * "오탐 0"인데, 소문자화는 그 보장을 깬다. 대소문자만 다른 질문은 임베딩
 * 경로로 넘어가 어차피 0.99 근처에서 히트한다.
 */
final class QuestionHash {

    private QuestionHash() {}

    /**
     * 앞뒤 공백을 제거하고 내부 연속 공백을 하나로 줄인 뒤 SHA-256 을 뜬다.
     *
     * <p>줄바꿈·탭·연속 스페이스만 정리하는 이유: 사용자가 같은 질문을 다시
     * 입력할 때 실제로 달라지는 건 대개 이것뿐이다. 그 이상 손대면 서로 다른
     * 질문이 같은 해시로 합쳐질 수 있다.
     *
     * @return 소문자 hex 64자
     */
    static String of(String question) {
        String normalized = normalize(question);
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(normalized.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException impossible) {
            // SHA-256 은 모든 JRE 가 반드시 제공한다.
            throw new IllegalStateException("SHA-256 을 쓸 수 없다", impossible);
        }
    }

    /**
     * 앞뒤 공백을 버리고 내부 연속 공백을 스페이스 하나로 만든다.
     *
     * <p>정규식을 쓰지 않는 이유가 둘 있다. Java 15 부터 문자열 안의 {@code \s} 는
     * 정규식 공백류가 아니라 <b>리터럴 스페이스 이스케이프</b>라서, 이스케이프를
     * 한 겹 빠뜨리면 컴파일은 통과하면서 탭·줄바꿈만 조용히 안 걸린다.
     * 그리고 {@link Character#isWhitespace}는 정규식 {@code \s} 보다 넓어
     * 유니코드 공백까지 잡는다.
     */
    static String normalize(String question) {
        if (question == null) {
            return "";
        }
        StringBuilder normalized = new StringBuilder(question.length());
        boolean pendingSpace = false;
        for (int i = 0; i < question.length(); i++) {
            char c = question.charAt(i);
            if (Character.isWhitespace(c)) {
                // 앞쪽 공백은 버린다. 뒤쪽 공백은 뒤에 글자가 안 나오면 그대로 버려진다.
                pendingSpace = !normalized.isEmpty();
            } else {
                if (pendingSpace) {
                    normalized.append(' ');
                    pendingSpace = false;
                }
                normalized.append(c);
            }
        }
        return normalized.toString();
    }
}
