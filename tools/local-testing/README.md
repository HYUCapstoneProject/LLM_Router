# 로컬 검증 도구

Anthropic API 키와 크레딧 없이 게이트웨이의 전체 경로를 돌려보기 위한 것들이다.
배포에는 쓰이지 않는다 — `scripts/deploy-backend.sh` 는 `backend/` 만 아카이브하므로
이 디렉터리는 VM 으로 올라가지 않는다.

호스트에 JDK 가 없어도 된다. 배포 스크립트와 같은 `eclipse-temurin:24-jdk` 컨테이너를 쓴다.

## mock_anthropic.py

Anthropic Messages API 목 서버. 응답 본문은 가짜지만 JSON 규격과 토큰 수는 진짜라
SDK 가 그대로 파싱한다.

```bash
python3 tools/local-testing/mock_anthropic.py &
# 게이트웨이에 ANTHROPIC_BASE_URL=http://127.0.0.1:8123 을 준다
```

요청에 실린 프롬프트를 stdout 에 찍으므로 "게이트웨이가 Claude 에 무엇을 보냈는지"를
직접 확인할 수 있다.

오류 분기(`GlobalExceptionHandler` 매핑)를 실제로 밟아보려면:

```bash
python3 tools/local-testing/mock_anthropic.py --fail quota         # → 402 LLM_QUOTA_EXCEEDED
python3 tools/local-testing/mock_anthropic.py --fail rate-limit    # → 429 LLM_RATE_LIMITED
python3 tools/local-testing/mock_anthropic.py --fail auth          # → 500 LLM_AUTH_FAILED
python3 tools/local-testing/mock_anthropic.py --fail unavailable   # → 503 LLM_UNAVAILABLE
python3 tools/local-testing/mock_anthropic.py --fail server-error  # → 503 LLM_UNAVAILABLE
```

`quota` 는 특히 챙겨둘 만하다. 게이트웨이가 402 를 판정하는 방식이
Anthropic 오류 메시지에 `credit balance` 가 들어있는지 보는 **문자열 매칭**이라
(`GlobalExceptionHandler` 주석에도 적혀 있다) 문구가 바뀌면 조용히 502 로 떨어진다.
이 모드가 그 분기를 반복 검증할 수 있는 유일한 수단이다.

> 목 서버를 끌 때 `pkill -f mock_anthropic` 은 쓰지 말 것. 그 명령을 실행한 셸의
> 명령줄에도 같은 문자열이 들어 있어 자기 자신을 죽인다. 포트로 찾는 편이 안전하다:
> ```bash
> kill $(ss -ltnpH 'sport = :8123' | grep -o 'pid=[0-9]*' | cut -d= -f2)
> ```

## StubVec.java

`StubRouterClient.embed()` 와 똑같은 벡터를 pgvector 리터럴로 찍는다.
캐시 히트 경로만 떼어내 검증할 때 쓴다 — 캐시 적재는 LLM 응답을 받은 뒤에 일어나므로,
미리 행을 심어두지 않으면 히트 경로를 볼 수 없다.

```bash
docker run --rm -v "$PWD/tools/local-testing:/w" -w /w eclipse-temurin:24-jdk \
  java StubVec.java "파이썬 데코레이터 설명해줘"
```

> ⚠️ `StubRouterClient` 의 알고리즘을 복제한 것이다. 그쪽 `embed()` 가 바뀌면
> 여기도 같이 고쳐야 한다. 안 그러면 "히트해야 하는데 안 한다"로 조용히 어긋난다.
> FastAPI 라우터가 붙어 stub 이 사라지면 이 파일도 같이 지운다.

## 전체 경로 한 번 돌려보기

```bash
# 1. 검증 전용 DB (실사용 llmrouter DB 를 건드리지 않는다)
docker exec llmrouter-postgres psql -U llmrouter -d postgres \
  -c "DROP DATABASE IF EXISTS localtest;" -c "CREATE DATABASE localtest;"
docker exec -i llmrouter-postgres psql -U llmrouter -d localtest \
  < infra/postgres/init/01-schema.sql

# 2. 목 서버
python3 tools/local-testing/mock_anthropic.py &

# 3. 게이트웨이 (호스트에 JDK 불필요)
PGPW=$(docker inspect llmrouter-postgres \
        --format '{{range .Config.Env}}{{println .}}{{end}}' \
        | sed -n 's/^POSTGRES_PASSWORD=//p')
docker run --rm --network host --user "$(id -u):$(id -g)" \
  -e ANTHROPIC_API_KEY=sk-ant-dummy \
  -e ANTHROPIC_BASE_URL=http://127.0.0.1:8123 \
  -e SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/localtest \
  -e POSTGRES_PASSWORD="$PGPW" \
  -e SERVER_PORT=8099 \
  -e GRADLE_USER_HOME=/gradle \
  -v "$PWD/backend:/workspace" -v "$HOME/.gradle-docker:/gradle" \
  -w /workspace eclipse-temurin:24-jdk bash ./gradlew bootRun --no-daemon

# 4. 확인
curl -s localhost:8099/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"자바 스트림 API 설명해줘","conversationId":"c1"}'
```

`GRADLE_USER_HOME` 을 리포지토리 밖(`~/.gradle-docker`)으로 빼는 이유:
컨테이너가 `--user` 로 돌기 때문에 기본 `~/.gradle` 을 못 쓰는데,
그렇다고 워크스페이스 안에 두면 Gradle 캐시가 리포지토리에 쌓인다.
