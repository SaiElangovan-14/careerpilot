import time

from app.services.llm import MODELS, get_client

for model in MODELS:
    start = time.time()
    try:
        text = get_client().models.generate_content(model=model, contents="say hi").text
        print(model, "OK", round(time.time() - start, 1), "s |", text.strip()[:40])
    except Exception as e:
        print(model, "FAILED", round(time.time() - start, 1), "s |", str(e)[:80])