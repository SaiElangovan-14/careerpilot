import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

MODELS = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]
ROUNDS = 3
BACKOFF_SECONDS = [5, 15]  # wait after round 1, then after round 2

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


def _generate_uncached(prompt: str, schema: type[BaseModel]) -> BaseModel:
    last_error = None
    out_of_quota: set[str] = set()

    for round_number in range(ROUNDS):
        for model in MODELS:
            if model in out_of_quota:
                continue
            try:
                response = get_client().models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0,
                    ),
                )
                if response.parsed is None:
                    raise ValueError("model returned output that does not match the schema")
                return response.parsed
            except Exception as e:
                last_error = e
                if _is_quota_error(e):
                    out_of_quota.add(model)

        if len(out_of_quota) == len(MODELS):
            break
        if round_number < ROUNDS - 1:
            time.sleep(BACKOFF_SECONDS[round_number])

    raise RuntimeError(f"All models failed: {last_error}")

import hashlib
import json
import os

CACHE_PATH = ".llm_cache.json"


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