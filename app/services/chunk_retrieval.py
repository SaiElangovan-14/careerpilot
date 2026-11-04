from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import ResumeChunk
from app.services.embeddings import embed
from app.services.retrieval import rrf_fuse


def _scope(stmt, resume_id: str, kinds):
    stmt = stmt.where(ResumeChunk.resume_id == resume_id)
    if kinds:
        stmt = stmt.where(ResumeChunk.kind.in_(kinds))
    return stmt


def vector_search_chunks(
    session: Session, resume_id: str, query: str, kinds=None, limit: int = 50
) -> list[int]:
    query_vec = embed([query])[0]
    stmt = select(ResumeChunk.id).order_by(
        ResumeChunk.embedding.cosine_distance(query_vec)
    ).limit(limit)
    return list(session.scalars(_scope(stmt, resume_id, kinds)))


def keyword_search_chunks(
    session: Session, resume_id: str, query: str, kinds=None, limit: int = 50
) -> list[int]:
    document = func.to_tsvector("english", ResumeChunk.title + " " + ResumeChunk.text)
    ts_query = func.websearch_to_tsquery("english", " or ".join(query.split()))
    stmt = (
        select(ResumeChunk.id)
        .where(document.op("@@")(ts_query))
        .order_by(func.ts_rank(document, ts_query).desc())
        .limit(limit)
    )
    return list(session.scalars(_scope(stmt, resume_id, kinds)))


def hybrid_search_chunks(
    session: Session,
    resume_id: str,
    query: str,
    top_k: int = 4,
    mode: str = "hybrid",
    kinds=None,
):
    rankings = []
    if mode in ("vector", "hybrid"):
        rankings.append(vector_search_chunks(session, resume_id, query, kinds))
    if mode in ("keyword", "hybrid"):
        rankings.append(keyword_search_chunks(session, resume_id, query, kinds))

    fused = rrf_fuse(rankings)[:top_k]
    ids = [chunk_pk for chunk_pk, _ in fused]
    rows = {
        c.id: c
        for c in session.scalars(select(ResumeChunk).where(ResumeChunk.id.in_(ids)))
    }
    return [(rows[chunk_pk], score) for chunk_pk, score in fused]