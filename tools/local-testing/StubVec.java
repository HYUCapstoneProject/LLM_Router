import java.util.Random;

/**
 * StubRouterClient.embed() 와 똑같은 벡터를 만들어 pgvector 리터럴로 찍는다.
 *
 * 왜 필요한가:
 * 캐시 히트 경로를 검증하려면 먼저 캐시에 행이 있어야 하는데,
 * 그러려면 LLM 을 한 번 호출해야 한다(캐시 적재가 응답을 받은 뒤에 일어나므로).
 * 이 도구로 벡터를 미리 만들어 cache_entry 에 직접 넣으면
 * API 키도 목 서버도 없이 히트 경로만 떼어내 볼 수 있다.
 *
 * ⚠️ StubRouterClient 의 알고리즘을 복제한 것이다.
 *    그쪽 embed() 가 바뀌면 여기도 같이 고쳐야 한다. 안 그러면
 *    "히트해야 하는데 안 한다"로 조용히 어긋난다.
 *
 * 사용:
 *   java tools/local-testing/StubVec.java "파이썬 데코레이터 설명해줘"
 *
 *   docker exec -i llmrouter-postgres psql -U llmrouter -d llmrouter -c \
 *     "INSERT INTO cache_entry
 *        (question, embedding, response, tier, model_id, embed_model_version, threshold)
 *      VALUES ('파이썬 데코레이터 설명해줘', '$(java StubVec.java '파이썬 데코레이터 설명해줘')'::vector,
 *              '미리 심어둔 답', 'medium', 'claude-sonnet-4-6', 'stub-hash-v0', 0.80);"
 */
public class StubVec {

    private static final int DIM = 1024;

    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("usage: java StubVec.java <message>");
            System.exit(2);
        }

        Random rnd = new Random(args[0].hashCode());

        float[] v = new float[DIM];
        double norm = 0.0;
        for (int i = 0; i < DIM; i++) {
            v[i] = (float) rnd.nextGaussian();
            norm += (double) v[i] * v[i];
        }
        norm = Math.sqrt(norm);

        StringBuilder sb = new StringBuilder(DIM * 12).append('[');
        for (int i = 0; i < DIM; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(Float.toString(v[i] / (float) norm));
        }
        System.out.println(sb.append(']'));
    }
}
