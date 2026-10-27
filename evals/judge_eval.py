import hashlib
import json
import os
import sys
import time

from app.services.judge import PROMPT, BulletVerdict, judge_bullet

CACHE_PATH = "evals/.judge_cache.json"


def _key(case: dict) -> str:
    raw = PROMPT + "\n" + case["source"] + "\n" + case["bullet"]
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def main(path: str = "evals/data/faithfulness.json") -> None:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)

    cache = _load_cache()
    tp = fp = fn = tn = 0  # "positive" = the judge flags the bullet as unfaithful
    skipped = 0
    quota_hit = False 

    for case in cases:
        key = _key(case)
        verdict = BulletVerdict(**cache[key]) if key in cache else None

        if verdict is None:
            for attempt in range(3):
                try:
                    verdict = judge_bullet(case["source"], case["bullet"])
                    break
                except RuntimeError as e:
                    print("retry", attempt + 1, "|", str(e)[:40])
                    if "429" in str(e):
                    quota_hit = True
                    break
                    time.sleep(30)
         if verdict is not None:
            cache[key] = verdict.model_dump()
            _save_cache(cache)
            time.sleep(5)

         if quota_hit and verdict is None:
            print("Quota exhausted, stopping. Run again later; cached cases are kept.")
            break

        if verdict is None:
            print("SKIP |", case["bullet"][:60])
            skipped += 1
            continue

        flagged = not verdict.supported
        unfaithful = case["unfaithful"]

        if flagged and unfaithful:
            tp += 1
        elif flagged and not unfaithful:
            fp += 1
        elif not flagged and unfaithful:
            fn += 1
        else:
            tn += 1

        ok = flagged == unfaithful
        print("OK  " if ok else "MISS", "|", case["bullet"][:70], "|", verdict.unsupported_claims)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_alarm = fp / (fp + tn) if fp + tn else 0.0
    print()
    print(f"cases={len(cases)} skipped={skipped} caught={tp} missed={fn} false_alarms={fp}")
    print(f"precision={precision:.2f} recall={recall:.2f} false_alarm_rate={false_alarm:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])