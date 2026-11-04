import hashlib

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import ResumeChunk
from app.services.chunking import Chunk
from app.services.embeddings import embed


def resume_id_for(resume_text: str) -> str:
    return hashlib.sha256(resume_text.encode("utf-8")).hexdigest()[:32]


def has_resume(session: Session, resume_id: str) -> bool:
    stmt = select(ResumeChunk.id).where(ResumeChunk.resume_id == resume_id).limit(1)
    return session.scalars(stmt).first() is not None


def ingest_chunks(session: Session, resume_id: str, chunks: list[Chunk]) -> int:
    if not chunks or has_resume(session, resume_id):
        return 0
    vectors = embed([c.text for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        session.add(
            ResumeChunk(
                resume_id=resume_id,
                chunk_id=chunk.chunk_id,
                kind=chunk.kind,
                title=chunk.title,
                text=chunk.text,
                embedding=vector,
            )
        )
    session.commit()
    return len(chunks)