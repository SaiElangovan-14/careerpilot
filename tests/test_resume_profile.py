from app.services import resume_profile
from app.services.llm_resume_parser import ResumeProfile

TEXT = "Sai Elangovan\nsai@example.com\nSkills: Python, FastAPI"


def fake_llm(text: str) -> ResumeProfile:
    return ResumeProfile(
        name="Sai",
        email="sai@example.com",
        skills=["Python"],
        education=[],
        projects=[],
        experience_summary=None,
    )


def test_uses_llm_when_it_works(monkeypatch):
    monkeypatch.setattr(resume_profile, "parse_resume_llm", fake_llm)
    profile, parser = resume_profile.build_profile(TEXT)
    assert parser == "llm"
    assert profile.skills == ["Python"]


def test_falls_back_to_regex_when_llm_fails(monkeypatch):
    def broken_llm(text: str) -> ResumeProfile:
        raise RuntimeError("LLM is down")

    monkeypatch.setattr(resume_profile, "parse_resume_llm", broken_llm)
    profile, parser = resume_profile.build_profile(TEXT)
    assert parser == "regex"
    assert "python" in profile.skills
    assert profile.email == "sai@example.com"