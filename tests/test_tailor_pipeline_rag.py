from app.services import tailor_pipeline
from app.services.llm_resume_parser import ResumeProfile
from app.services.tailor import TailoredBullet, TailoredResume
from app.services.tailor_check import BulletCheck


def make_profile():
    return ResumeProfile(
        name="Sai",
        email=None,
        skills=["Python", "FastAPI"],
        education=[],
        projects=[],
        experience_summary=None,
    )


def test_tailor_and_verify_rag_uses_chunks_and_profile_skills(monkeypatch):
    seen = {}

    def fake_rag(chunks, skills, job_title, job_description):
        seen.update(chunks=chunks, skills=skills, title=job_title)
        return TailoredResume(
            summary="s",
            bullets=[
                TailoredBullet(
                    text="Built a backend", source="TenantCore: Built a backend"
                )
            ],
            matched_skills=["Python"],
            missing_skills=[],
        )

    def fake_verify(profile, source, text, missing_skills):
        return BulletCheck(
            text=text,
            source=source,
            grounded=True,
            claims_missing_skill=[],
            unsupported_numbers=[],
            unsupported_claims=[],
            repaired=False,
            ok=True,
        )

    monkeypatch.setattr(tailor_pipeline, "tailor_resume_rag", fake_rag)
    monkeypatch.setattr(tailor_pipeline, "_verify", fake_verify)

    report = tailor_pipeline.tailor_and_verify_rag(
        make_profile(), ["chunk"], "Backend Intern", "Build APIs"
    )

    assert seen["chunks"] == ["chunk"]
    assert seen["skills"] == ["Python", "FastAPI"]
    assert [b.text for b in report.bullets] == ["Built a backend"]
    assert report.rejected == []