from sqlalchemy.orm import Session

from app.services.chunk_retrieval import hybrid_search_chunks

CONTEXT_KINDS = ("project", "experience")
FIT_KINDS = CONTEXT_KINDS + ("tech",)  # fit analysis also needs the technology lists


def retrieve_context(
    session: Session,
    resume_id: str,
    job_title: str,
    job_description: str,
    top_k: int = 6,
    kinds: tuple[str, ...] = CONTEXT_KINDS,
):
    query = job_title + " " + job_description[:1500]
    results = hybrid_search_chunks(
        session,
        resume_id,
        query,
        top_k=top_k,
        mode="hybrid",
        kinds=kinds,
    )
    return [chunk for chunk, _score in results]


def format_context(chunks) -> str:
    return "\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)