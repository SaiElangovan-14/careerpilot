import json

from sqlalchemy import delete

from app.database import ResumeChunk, SessionLocal
from app.services.chunk_retrieval import hybrid_search_chunks
from app.services.chunking import chunk_profile
from app.services.ingest import ingest_chunks, resume_id_for
from app.services.llm_resume_parser import ResumeProfile

DATA_PATH = "evals/data/retrieval_cases.json"
MODES = ("keyword", "vector", "hybrid")
K = 5
KINDS = ("project", "experience")


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    for position, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def main() -> None:
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    profile = ResumeProfile.model_validate(data["resume"])
    chunks = chunk_profile(profile)

    known = {c.chunk_id for c in chunks}
    for case in data["cases"]:
        missing = set(case["relevant"]) - known
        if missing:
            raise SystemExit(f"{case['title']}: unknown chunk ids {sorted(missing)}")

    rid = "eval-" + resume_id_for(profile.model_dump_json())
    totals = {m: {"recall": 0.0, "mrr": 0.0} for m in MODES}

    with SessionLocal() as session:
        session.execute(delete(ResumeChunk).where(ResumeChunk.resume_id == rid))
        session.commit()
        ingest_chunks(session, rid, chunks)

        for case in data["cases"]:
            query = case["title"] + " " + case["description"]
            relevant = case["relevant"]
            print(f"\n{case['title']}  relevant={relevant}")
            for mode in MODES:
                results = hybrid_search_chunks(
                    session, rid, query, top_k=K, mode=mode, kinds=KINDS
                )
                ranked = [c.chunk_id for c, _ in results]
                recall = recall_at_k(ranked, relevant, K)
                rr = reciprocal_rank(ranked, relevant)
                totals[mode]["recall"] += recall
                totals[mode]["mrr"] += rr
                print(f"  {mode:8} recall@{K}={recall:.2f} rr={rr:.2f} {ranked}")

    n = len(data["cases"])
    print("\nSUMMARY")
    for mode in MODES:
        r = totals[mode]["recall"] / n
        m = totals[mode]["mrr"] / n
        print(f"  {mode:8} recall@{K}={r:.2f} mrr={m:.2f}")


if __name__ == "__main__":
    main()