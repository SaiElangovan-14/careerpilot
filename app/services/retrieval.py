from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import Job
from app.services.embeddings import embed


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def vector_search(session: Session, query: str, limit: int = 50) -> list[int]:
    query_vec = embed([query])[0]
    stmt = select(Job.id).order_by(Job.embedding.cosine_distance(query_vec)).limit(limit)
    return list(session.scalars(stmt))


def keyword_search(session: Session, query: str, limit: int = 50) -> list[int]:
    document = func.to_tsvector("english", Job.title + " " + Job.description)
    ts_query = func.websearch_to_tsquery("english", " or ".join(query.split()))
    stmt = (
        select(Job.id)
        .where(document.op("@@")(ts_query))
        .order_by(func.ts_rank(document, ts_query).desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def hybrid_search(session: Session, query: str, top_k: int = 5, mode: str = "hybrid"):
    rankings = []
    if mode in ("vector", "hybrid"):
        rankings.append(vector_search(session, query))
    if mode in ("keyword", "hybrid"):
        rankings.append(keyword_search(session, query))

    fused = rrf_fuse(rankings)[:top_k]
    ids = [job_id for job_id, _ in fused]
    jobs = {job.id: job for job in session.scalars(select(Job).where(Job.id.in_(ids)))}
    return [(jobs[job_id], score) for job_id, score in fused]