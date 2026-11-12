import pytest
from fastapi import HTTPException

from app.ratelimit import RateLimiter


def test_blocks_after_the_limit_and_recovers_after_the_window():
    limiter = RateLimiter(limit=2, window_seconds=60)

    limiter.check("a", now=0)
    limiter.check("a", now=1)
    with pytest.raises(HTTPException) as e:
        limiter.check("a", now=2)
    assert e.value.status_code == 429
    assert int(e.value.headers["Retry-After"]) > 0

    limiter.check("a", now=61)  # the first call has left the window


def test_each_client_has_its_own_limit():
    limiter = RateLimiter(limit=1, window_seconds=60)

    limiter.check("a", now=0)
    limiter.check("b", now=0)  # a different client is not affected