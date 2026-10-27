from app.services.cover_letter import select_projects
from app.services.llm_resume_parser import Project, ResumeProfile


def _profile():
    return ResumeProfile.model_construct(
        projects=[
            Project(
                name="Photo Gallery",
                description="Static website with an image carousel",
                technologies=["HTML", "CSS"],
            ),
            Project(
                name="Job Board API",
                description="REST API for employers and candidates",
                technologies=["FastAPI", "PostgreSQL", "Docker"],
            ),
        ]
    )


def test_most_relevant_project_comes_first():
    chosen = select_projects(
        _profile(), "Backend Intern", "Build REST APIs with FastAPI and PostgreSQL"
    )
    assert chosen[0].name == "Job Board API"


def test_irrelevant_projects_are_excluded():
    chosen = select_projects(
        _profile(), "Backend Intern", "Build REST APIs with FastAPI and PostgreSQL"
    )
    assert [p.name for p in chosen] == ["Job Board API"]


def test_respects_k_and_returns_empty_when_nothing_matches():
    profile = _profile()
    assert len(select_projects(profile, "Dev", "REST API HTML CSS", k=1)) == 1
    assert select_projects(profile, "Chef", "Cooking pasta in a kitchen") == []