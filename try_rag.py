from sqlalchemy import select

from app.database import Job, SessionLocal
from app.services.chunk_retrieval import hybrid_search_chunks
from app.services.chunking import chunk_profile
from app.services.ingest import ingest_chunks, resume_id_for
from app.services.llm_resume_parser import Education, Project, ResumeProfile

profile = ResumeProfile(
    name="Sai",
    email=None,
    skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
    education=[
        Education(
            degree="B.E. Electronics and Communication Engineering",
            institution="Rajalakshmi Institute of Technology",
            year="2026",
        )
    ],
    projects=[
        Project(
            name="TenantCore",
            description="Built a multi-tenant SaaS backend with FastAPI and PostgreSQL",
            technologies=["FastAPI", "PostgreSQL", "Docker"],
        ),
        Project(
            name="Job Board API",
            description="Built a REST API for job listings with authentication, deployed with Docker",
            technologies=["FastAPI", "PostgreSQL", "Docker"],
        ),
        Project(
            name="CareerPilot",
            description="AI job-application copilot with hybrid retrieval over pgvector, LLM resume tailoring and a faithfulness judge",
            technologies=["Python", "pgvector", "Gemini"],
        ),
    ],
    experience_summary=None,
)

rid = resume_id_for(profile.model_dump_json())

with SessionLocal() as session:
    print("ingested chunks:", ingest_chunks(session, rid, chunk_profile(profile)))
    jobs = list(session.scalars(select(Job).limit(4)))
    for job in jobs:
        print("\nJOB:", job.title)
        query = job.title + " " + job.description[:1500]
        for mode in ("keyword", "vector", "hybrid"):
            results = hybrid_search_chunks(session, rid, query, top_k=3, mode=mode)
            print(" ", mode, [(c.chunk_id, round(score, 3)) for c, score in results])