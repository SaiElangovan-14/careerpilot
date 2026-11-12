import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class RateLimiter:
    """Allow at most `limit` calls per `window_seconds` for each key."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        calls = self.hits[key]
        while calls and now - calls[0] >= self.window:
            calls.popleft()
        if len(calls) >= self.limit:
            retry_after = int(self.window - (now - calls[0])) + 1
            raise HTTPException(
                status_code=429,
                detail="Too many requests, slow down",
                headers={"Retry-After": str(retry_after)},
            )
        calls.append(now)


ai_limiter = RateLimiter(limit=10, window_seconds=60)


def limit_ai(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    ai_limiter.check(client)