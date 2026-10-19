import json
import sys

from app.services.llm_resume_parser import parse_resume_llm
from app.services.resume_parser import extract_pdf_text, parse_resume_text


def norm(skills: list[str]) -> set[str]:
    return {s.strip().lower() for s in skills}


def score(found: set[str], gold: set[str]) -> tuple[float, float]:
    correct = len(found & gold)
    precision = correct / len(found) if found else 0.0
    recall = correct / len(gold) if gold else 0.0
    return precision, recall


def main(pdf_path: str, gold_path: str) -> None:
    with open(pdf_path, "rb") as f:
        text = extract_pdf_text(f.read())
    with open(gold_path, encoding="utf-8") as f:
        gold = norm(json.load(f)["skills"])

    regex_found = norm(parse_resume_text(text)["skills"])
    llm_found = norm(parse_resume_llm(text).skills)

    print(f"gold skills: {len(gold)}\n")
    print(f"{'parser':<8} {'found':>6} {'precision':>10} {'recall':>8}")
    for name, found in (("regex", regex_found), ("llm", llm_found)):
        precision, recall = score(found, gold)
        print(f"{name:<8} {len(found):>6} {precision:>10.3f} {recall:>8.3f}")

    print("\nLLM extra (not in gold):", sorted(llm_found - gold))
    print("LLM missed:", sorted(gold - llm_found))
    print("Regex missed:", sorted(gold - regex_found))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m evals.parser_eval <resume.pdf> <gold.json>")
    main(sys.argv[1], sys.argv[2])