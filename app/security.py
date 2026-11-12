import hmac
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = os.getenv("CAREERPILOT_API_KEY")
    if not expected:
        # fail closed: a server without a configured key must not run open
        raise HTTPException(status_code=503, detail="Server API key is not configured")
    if not key or not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")