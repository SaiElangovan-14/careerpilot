from sqlalchemy.orm import Session

from app.services.chunking import CHUNKER_VERSION, chunk_profile
from app.services.ingest import ingest_chunks, resume_id_for
from app.services.llm_resume_parser import ResumeProfile
from app.services.rag_context import retrieve_context


def build_context(session: Session, profile: ResumeProfile, job, top_k: int = 6):
    resume_id = resume_id_for(f"v{CHUNKER_VERSION}:" + profile.model_dump_json())
    ingest_chunks(session, resume_id, chunk_profile(profile))
    return retrieve_context(
        session, resume_id, job.title, job.description, top_k=top_k
    )