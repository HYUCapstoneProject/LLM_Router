# ── 1단계: 빌드 ──────────────────────────────────────────────────────
FROM eclipse-temurin:24-jdk AS build

WORKDIR /build

# 의존성 파일만 먼저 복사한다.
# src 가 바뀌어도 의존성 레이어는 캐시에서 재사용되어 빌드가 빨라진다.
COPY gradlew settings.gradle build.gradle gradle.properties ./
COPY gradle ./gradle
RUN chmod +x gradlew && ./gradlew dependencies --no-daemon || true

COPY src ./src
RUN ./gradlew bootJar --no-daemon -x test

# ── 2단계: 실행 ──────────────────────────────────────────────────────
# JRE 만 담아 이미지를 줄인다.
# eclipse-temurin:24-jre 를 못 찾으면 24-jdk 로 바꿔도 동작한다(용량만 커짐).
FROM eclipse-temurin:24-jre

WORKDIR /app

# root 로 돌리지 않는다
RUN groupadd -r app && useradd -r -g app app

COPY --from=build /build/build/libs/*.jar app.jar
RUN chown app:app /app/app.jar

USER app
EXPOSE 8080

# GCE e2-standard-2(8GB)에 PostgreSQL 과 공동 배치되므로 힙을 절반으로 제한한다.
# 제한하지 않으면 JVM 이 컨테이너 메모리 대부분을 잡아 Postgres 와 충돌한다.
ENTRYPOINT ["java", "-XX:MaxRAMPercentage=50", "-jar", "/app/app.jar"]
