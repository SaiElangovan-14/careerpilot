from app.services import tailor_pipeline
from app.services.judge import BulletVerdict
from app.services.llm_resume_parser import Project, ResumeProfile
from app.services.tailor import TailoredBullet, TailoredResume

PROFILE = ResumeProfile(
    name=None,
    email=None,
    skills=["python", "fastapi", "postgresql"],
    education=[],
    projects=[
        Project(
            name="Job Board API",
            description="Built a REST API with FastAPI and PostgreSQL",
            technologies=["fastapi", "postgresql"],
        )
    ],
    experience_summary=None,
)


def _supported(source, bullet):
    return BulletVerdict(supported=True, unsupported_claims=[])


def test_cheating_bullets_are_rejected(monkeypatch):
    fake = TailoredResume(
        summary="x",
        bullets=[
            TailoredBullet(
                text="Built a REST API with FastAPI and PostgreSQL.",
                source="Built a REST API with FastAPI and PostgreSQL",
            ),
            TailoredBullet(
                text="Ran Kubernetes clusters in production.",
                source="Managed Kubernetes clusters at Amazon",
            ),
        ],
        matched_skills=["fastapi"],
        missing_skills=["kubernetes"],
    )
    monkeypatch.setattr(tailor_pipeline, "tailor_resume", lambda *a, **k: fake)
    monkeypatch.setattr(tailor_pipeline, "judge_bullet", _supported)

    report = tailor_pipeline.tailor_and_verify(PROFILE, "Intern", "FastAPI, Kubernetes")

    assert len(report.bullets) == 1
    assert len(report.rejected) == 1
    assert report.rejected[0].claims_missing_skill == ["kubernetes"]


def test_judge_rejects_stretched_bullet(monkeypatch):
    fake = TailoredResume(
        summary="x",
        bullets=[
            TailoredBullet(
                text="Built a REST API with FastAPI for big data processing.",
                source="Built a REST API with FastAPI and PostgreSQL",
            ),
        ],
        matched_skills=["fastapi"],
        missing_skills=[],
    )
    monkeypatch.setattr(tailor_pipeline, "tailor_resume", lambda *a, **k: fake)
    monkeypatch.setattr(
        tailor_pipeline,
        "judge_bullet",
        lambda source, bullet: BulletVerdict(
            supported=False, unsupported_claims=["big data processing"]
        ),
    )

    report = tailor_pipeline.tailor_and_verify(PROFILE, "Intern", "FastAPI")

    assert report.bullets == []
    assert report.rejected[0].unsupported_claims == ["big data processing"]