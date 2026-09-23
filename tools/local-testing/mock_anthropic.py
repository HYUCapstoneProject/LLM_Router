#!/usr/bin/env python3
"""Anthropic Messages API 목 서버.

실제 API 키와 크레딧 없이 게이트웨이의 전체 경로(캐시 미스 → LLM 호출 → 캐시 적재
→ request_log 적재)를 돌려보기 위한 것이다. 응답 본문은 가짜지만 토큰 수와
JSON 형태는 SDK 가 파싱할 수 있는 진짜 규격이다.

사용:
    python3 tools/local-testing/mock_anthropic.py &
    # 게이트웨이에 ANTHROPIC_BASE_URL=http://127.0.0.1:8123 을 준다

옵션:
    --port     기본 8123
    --fail     오류 분기 검증용. GlobalExceptionHandler 매핑을 실제로 밟아볼 수 있다.
               quota | rate-limit | auth | unavailable | server-error

주의: 이 서버는 요청에 실린 프롬프트를 그대로 stdout 에 찍는다.
게이트웨이가 Claude 에 무엇을 보냈는지 확인하는 용도이므로,
실제 사용자 데이터가 흐르는 환경에서는 절대 쓰지 않는다.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# --fail 값 → (HTTP 상태, Anthropic 오류 타입, 메시지)
# 메시지 문구는 게이트웨이 GlobalExceptionHandler 의 분기 조건과 맞춰야 한다.
# 특히 quota 는 "credit balance" 문자열 매칭으로 402 를 판정한다.
FAILURES = {
    "quota": (400, "invalid_request_error",
              "Your credit balance is too low to access the Anthropic API."),
    "rate-limit": (429, "rate_limit_error", "Number of requests has exceeded your rate limit."),
    "auth": (401, "authentication_error", "invalid x-api-key"),
    "unavailable": (500, "api_error", "Internal server error"),
    "server-error": (529, "overloaded_error", "Overloaded"),
}


def make_handler(fail_mode: str):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")

            model = body.get("model", "unknown")
            print(f"[mock] model={model}", flush=True)
            for msg in body.get("messages", []):
                content = msg.get("content")
                text = content if isinstance(content, str) else \
                    " ".join(part.get("text", "") for part in content)
                print(f"[mock] {msg.get('role')}: {text}", flush=True)

            if fail_mode:
                status, err_type, message = FAILURES[fail_mode]
                self._send(status, {"type": "error",
                                    "error": {"type": err_type, "message": message}})
                return

            self._send(200, {
                "id": "msg_mock_0001",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": f"[mock:{model}] 목 서버 응답입니다."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                # 티어별 비용 차이를 눈으로 구분할 수 있게 모델마다 다른 값을 준다.
                "usage": {
                    "input_tokens": 100 + len(model),
                    "output_tokens": 200 + len(model),
                },
            })

        def _send(self, status: int, payload: dict):
            out = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass   # 기본 액세스 로그는 끈다. 위에서 필요한 것만 찍는다.

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--fail", choices=sorted(FAILURES), default=None)
    args = parser.parse_args()

    mode = f" (fail={args.fail})" if args.fail else ""
    print(f"[mock] listening on http://127.0.0.1:{args.port}{mode}", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(args.fail)).serve_forever()


if __name__ == "__main__":
    main()
