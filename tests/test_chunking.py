from app.services.chunking import chunk_profile
from app.services.llm_resume_parser import Education, Project, ResumeProfile


def make_profile():
    return ResumeProfile(
        name="Sai",
        email=None,
        skills=["Python", "FastAPI"],
        education=[Education(degree="B.E. CSE", institution="Some College", year="2027")],
        projects=[
            Project(
                name="TenantCore",
                description="Built a multi-tenant SaaS backend",
                technologies=["FastAPI", "PostgreSQL"],
            ),
            Project(name="Blog", description="A simple blog", technologies=[]),
        ],
        experience_summary=None,
    )


def test_one_chunk_per_project_with_ids():
    chunks = chunk_profile(make_profile())
    projects = [c for c in chunks if c.kind == "project"]
    assert [c.chunk_id for c in projects] == ["project-0", "project-1"]
    assert "TenantCore" in projects[0].text
    assert "PostgreSQL" in projects[0].text


def test_skills_education_chunks_and_no_empty_experience():
    chunks = chunk_profile(make_profile())
    kinds = [c.kind for c in chunks]
    assert "skills" in kinds
    assert "education" in kinds
    assert "experience" not in kinds


def test_chunk_ids_unique():
    chunks = chunk_profile(make_profile())
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))