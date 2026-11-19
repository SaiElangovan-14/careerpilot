import os
import time
import logging

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

MODELS = ["gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.7-flash", "gemini-3.8-flash"]
ROUNDS = 3
BACKOFF_SECONDS = [5, 15]  # wait after round 1, then after round 2
DEADLINE_SECONDS = float(os.getenv("LLM_DEADLINE_SECONDS", "75"))  # stop trying new attempts after this many seconds

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=60_000),  # milliseconds
        )
    return _client

def _is_quota_error(error: Exception) -> bool:
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _is_timeout_error(error: Exception) -> bool:
    text = str(error)
    return "504" in text or "DEADLINE_EXCEEDED" in text

class LLMAuthError(RuntimeError):
    """The API key was rejected. Retrying cannot fix this."""


def _is_auth_error(error: Exception) -> bool:
    code = getattr(error, "code", None)
    text = str(error)
    return code in (401, 403) or "UNAUTHENTICATED" in text or "PERMISSION_DENIED" in text

logger = logging.getLogger("careerpilot.llm")


def _generate_uncached(prompt: str, schema: type[BaseModel]) -> BaseModel:
    last_error = None
    skipped: set[str] = set()  # out of quota or timing out: not worth retrying this request

    extra = {}
    budget = os.getenv("LLM_THINKING_BUDGET")
    if budget is not None:
        extra["thinking_config"] = types.ThinkingConfig(thinking_budget=int(budget))
    overall_start = time.monotonic()

    for round_number in range(ROUNDS):
        for model in MODELS:
            if model in skipped:
                continue
            if last_error is not None and time.monotonic() - overall_start >= DEADLINE_SECONDS:
                raise RuntimeError(f"LLM deadline exceeded: {last_error}")
            started = time.perf_counter()
            try:
                response = get_client().models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0,
                        **extra,
                    ),
                )
                if response.parsed is None:
                    raise ValueError("model returned output that does not match the schema")
                logger.info(
                    "llm model=%s round=%d ok %.0fms",
                    model, round_number + 1, (time.perf_counter() - started) * 1000,
                )
                return response.parsed
            except Exception as e:
                last_error = e
                if _is_auth_error(e):
                    raise LLMAuthError(f"Gemini rejected the API key: {str(e)[:200]}") from e
                logger.warning(
                    "llm model=%s round=%d failed %.0fms %s: %s",
                    model, round_number + 1, (time.perf_counter() - started) * 1000,
                    type(e).__name__, str(e)[:100],
                )
                if _is_quota_error(e) or _is_timeout_error(e):
                    skipped.add(model)

        if len(skipped) == len(MODELS):
            break
        if round_number < ROUNDS - 1:
            time.sleep(BACKOFF_SECONDS[round_number])

    raise RuntimeError(f"All models failed: {last_error}")

import hashlib
import json
import os

CACHE_PATH = os.getenv("LLM_CACHE_PATH", ".llm_cache.json")


def _cache_load() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _cache_save(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def generate_structured(prompt, schema):
    if os.getenv("LLM_CACHE", "1") != "1":
        return _generate_uncached(prompt, schema)

    raw = schema.__name__ + json.dumps(schema.model_json_schema(), sort_keys=True) + prompt
    key = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    cache = _cache_load()
    if key in cache:
        return schema.model_validate(cache[key])

    result = _generate_uncached(prompt, schema)
    cache[key] = result.model_dump()
    _cache_save(cache)
    return result