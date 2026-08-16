package com.llmrouter.service;

import java.util.StringJoiner;

/**
 * float[] → pgvector 리터럴 변환.
 *
 * cache_entry 와 escalation_log 두 곳이 vector(1024) 컬럼을 쓴다.
 * 양쪽이 각자 포맷하면 언젠가 미묘하게 어긋나므로 한 곳에 둔다.
 */
final class PgVector {

    private PgVector() {}

    /** "[0.1,0.2,...]" */
    static String literal(float[] v) {
        StringJoiner sj = new StringJoiner(",", "[", "]");
        for (float f : v) {
            sj.add(Float.toString(f));
        }
        return sj.toString();
    }
}
