from types import SimpleNamespace

from app.services import rag_pipeline
from app.services.llm_resume_parser import Project, ResumeProfile


def make_profile():
    return ResumeProfile(
        name="Sai",
        email=None,
        skills=["Python"],
        education=[],
        projects=[
            Project(
                name="TenantCore",
                description="Built a backend. Added JWT auth.",
                technologies=["FastAPI"],
            )
        ],
        experience_summary=None,
    )


def test_build_context_ingests_then_retrieves_with_same_resume_id(monkeypatch):
    calls = []

    def fake_ingest(session, resume_id, chunks):
        calls.append(("ingest", resume_id, [c.chunk_id for c in chunks]))
        return len(chunks)

    def fake_retrieve(session, resume_id, title, description, top_k, **kwargs):
        calls.append(("retrieve", resume_id, title, top_k))
        return ["chunk"]

    monkeypatch.setattr(rag_pipeline, "ingest_chunks", fake_ingest)
    monkeypatch.setattr(rag_pipeline, "retrieve_context", fake_retrieve)

    job = SimpleNamespace(title="Backend Intern", description="Build APIs")
    result = rag_pipeline.build_context(None, make_profile(), job)

    assert result == ["chunk"]
    assert calls[0][0] == "ingest"
    assert calls[1][0] == "retrieve"
    assert calls[0][1] == calls[1][1]
    assert calls[0][2] == ["project-0-0", "project-0-1", "project-0-tech", "skills"]
    assert calls[1][2:] == ("Backend Intern", 6)

def test_build_context_passes_kinds_through(monkeypatch):
    from types import SimpleNamespace

    from app.services import rag_pipeline

    seen = {}

    def fake_retrieve(session, resume_id, title, description, top_k, kinds):
        seen["kinds"] = kinds
        return []

    monkeypatch.setattr(rag_pipeline, "chunk_profile", lambda profile: [])
    monkeypatch.setattr(rag_pipeline, "ingest_chunks", lambda session, resume_id, chunks: None)
    monkeypatch.setattr(rag_pipeline, "retrieve_context", fake_retrieve)

    profile = SimpleNamespace(model_dump_json=lambda: "{}")
    job = SimpleNamespace(title="Backend Intern", description="Needs APIs")

    rag_pipeline.build_context(None, profile, job)
    assert seen["kinds"] == rag_pipeline.CONTEXT_KINDS  # default: no tech chunks

    rag_pipeline.build_context(None, profile, job, kinds=("project", "tech"))
    assert seen["kinds"] == ("project", "tech")