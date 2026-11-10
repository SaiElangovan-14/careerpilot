import time

from google.genai import types

from app.services.llm import MODELS, get_client

for model in MODELS:
    start = time.time()
    try:
        response = get_client().models.generate_content(
            model=model,
            contents="Reply with the single word: ok",
            config=types.GenerateContentConfig(temperature=0),
        )
        print(model, "OK", round(time.time() - start, 1), "s:", response.text.strip()[:40])
    except Exception as e:
        print(model, "FAILED", round(time.time() - start, 1), "s:", str(e)[:120])