import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

MODELS = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]
RETRIES_PER_MODEL = 1

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=60_000),  # milliseconds
        )
    return _client

def generate_structured(prompt: str, schema: type[BaseModel]) -> BaseModel:
    last_error = None
    for model in MODELS:
        for attempt in range(RETRIES_PER_MODEL):
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
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"All models failed: {last_error}")