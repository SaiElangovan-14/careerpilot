from app.services import cover_letter
from app.services.cover_letter import (
    CoverLetterBody,
    LetterSentence,
    select_projects,
)
from app.services.llm_resume_parser import Project, ResumeProfile
from app.services.tailor_check import BulletCheck


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

def test_draft_puts_projects_in_prompt_and_returns_body(monkeypatch):
    seen = {}

    def fake_llm(prompt, schema):
        seen["prompt"] = prompt
        return CoverLetterBody(
            sentences=[LetterSentence(text="I built a REST API.", source="REST API")]
        )

    monkeypatch.setattr(cover_letter, "generate_structured", fake_llm)
    projects = _profile().projects
    body = cover_letter.draft_cover_letter_body(projects, "Backend Intern", "FastAPI")

    assert "Job Board API" in seen["prompt"]
    assert "untrusted" in seen["prompt"]
    assert body.sentences[0].source == "REST API"

def test_letter_keeps_only_verified_sentences(monkeypatch):
    def fake_draft(projects, job_title, job_description):
        return CoverLetterBody(
            sentences=[
                LetterSentence(text="I built a REST API with FastAPI.", source="REST API"),
                LetterSentence(text="I deployed it on Kubernetes.", source="REST API"),
            ]
        )

    def fake_verify(profile, source, text, missing_skills):
        return BulletCheck(
            text=text,
            source=source,
            grounded=True,
            claims_missing_skill=[],
            unsupported_numbers=[],
            ok="Kubernetes" not in text,
        )

    monkeypatch.setattr(cover_letter, "draft_cover_letter_body", fake_draft)
    monkeypatch.setattr(cover_letter, "_verify", fake_verify)

    report = cover_letter.draft_and_verify_letter(
        _profile(), "Backend Intern", "Build REST APIs with FastAPI"
    )

    assert report.body == "I built a REST API with FastAPI."
    assert len(report.kept) == 1
    assert len(report.rejected) == 1


def test_letter_is_empty_when_no_project_matches(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("LLM must not be called when nothing matches")

    monkeypatch.setattr(cover_letter, "draft_cover_letter_body", boom)
    report = cover_letter.draft_and_verify_letter(
        _profile(), "Chef", "Cooking pasta in a kitchen"
    )
    assert report.body == ""
    assert report.kept == [] and report.rejected == []