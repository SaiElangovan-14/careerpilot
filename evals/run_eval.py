import json
import sys
from pathlib import Path

from sqlalchemy import text

from app.database import Job, SessionLocal, engine, init_db
from app.services.embeddings import embed
from app.services.retrieval import hybrid_search
from evals.metrics import hit_rate_at_k, recall_at_k, reciprocal_rank

MODES = ("keyword", "vector", "hybrid")
K = 5


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main(path: str) -> None:
    if "eval" not in (engine.url.database or ""):
        sys.exit("Refusing to run: database name must contain 'eval' (this script wipes the jobs table).")

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs = data["jobs"]
    queries = data["queries"]

    init_db()
    vectors = embed([f"{j['title']} {j['description']}" for j in jobs])

    with SessionLocal() as session:
        session.execute(text("TRUNCATE jobs RESTART IDENTITY"))
        rows = [
            Job(
                title=j["title"],
                company=j.get("company", ""),
                description=j["description"],
                embedding=v,
            )
            for j, v in zip(jobs, vectors)
        ]
        session.add_all(rows)
        session.flush()
        key_to_id = {j["key"]: row.id for j, row in zip(jobs, rows)}
        session.commit()

        print(f"{len(jobs)} jobs, {len(queries)} queries, k={K}\n")
        print(f"{'mode':<10} {'recall@' + str(K):>10} {'hit@' + str(K):>8} {'MRR':>8}")

        for mode in MODES:
            recalls, hits, rrs = [], [], []
            for q in queries:
                relevant = {key_to_id[k] for k in q["relevant"]}
                results = hybrid_search(session, q["query"], top_k=K, mode=mode)
                ids = [job.id for job, _score in results]
                recalls.append(recall_at_k(ids, relevant, K))
                hits.append(hit_rate_at_k(ids, relevant, K))
                rrs.append(reciprocal_rank(ids, relevant))
            print(f"{mode:<10} {mean(recalls):>10.3f} {mean(hits):>8.3f} {mean(rrs):>8.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m evals.run_eval <golden.json>")
    main(sys.argv[1])