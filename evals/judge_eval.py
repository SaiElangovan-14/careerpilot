import json
import sys
import time

from app.services.judge import judge_bullet


def main(path: str = "evals/data/faithfulness.json") -> None:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)

    tp = fp = fn = tn = 0  # "positive" = the judge flags the bullet as unfaithful
    skipped = 0
    for case in cases:
        verdict = None
        for attempt in range(3):
            try:
                verdict = judge_bullet(case["source"], case["bullet"])
                break
            except RuntimeError as e:
                print("retry", attempt + 1, "|", str(e)[:40])
                time.sleep(15)
        if verdict is None:
            print("SKIP |", case["bullet"][:60])
            skipped += 1
            continue
        
        flagged = not verdict.supported
        unfaithful =  case["unfaithful"]

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
    print(f"cases={len(cases)} skipped={skipped} caught={tp}  missed={fn}  false_alarms={fp}")
    print(f"precision={precision:.2f}  recall={recall:.2f}  false_alarm_rate={false_alarm:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])