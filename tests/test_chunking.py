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


def test_projects_split_into_sentence_chunks():
    chunks = chunk_profile(make_profile())
    project_chunks = [c for c in chunks if c.kind == "project"]
    assert [c.chunk_id for c in project_chunks] == [
        "project-0-0",
        "project-0-tech",
        "project-1-0",
    ]
    assert project_chunks[0].text == "TenantCore: Built a multi-tenant SaaS backend"
    assert "PostgreSQL" in project_chunks[1].text


def test_multi_sentence_description_splits():
    profile = make_profile()
    profile.projects[1].description = "Built a blog. Added comments and auth."
    ids = [c.chunk_id for c in chunk_profile(profile) if c.kind == "project"]
    assert "project-1-0" in ids
    assert "project-1-1" in ids

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