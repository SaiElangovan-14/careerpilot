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

GOOD = "Built a REST API with FastAPI and PostgreSQL"
STRETCH = "Built a REST API with FastAPI for big data processing."


def _supported(source, bullet):
    return BulletVerdict(supported=True, unsupported_claims=[])


def _judge_by_content(source, bullet):
    if "big data" in bullet:
        return BulletVerdict(supported=False, unsupported_claims=["big data processing"])
    return BulletVerdict(supported=True, unsupported_claims=[])


def _tailored(*bullets, missing=None):
    return TailoredResume(
        summary="x",
        bullets=list(bullets),
        matched_skills=["fastapi"],
        missing_skills=missing or [],
    )


def test_invented_source_is_rejected_without_repair(monkeypatch):
    fake = _tailored(
        TailoredBullet(text=GOOD + ".", source=GOOD),
        TailoredBullet(
            text="Ran Kubernetes clusters in production.",
            source="Managed Kubernetes clusters at Amazon",
        ),
        missing=["kubernetes"],
    )
    monkeypatch.setattr(tailor_pipeline, "tailor_resume", lambda *a, **k: fake)
    monkeypatch.setattr(tailor_pipeline, "judge_bullet", _supported)
    monkeypatch.setattr(tailor_pipeline, "rewrite_bullet", lambda *a, **k: "unused")

    report = tailor_pipeline.tailor_and_verify(PROFILE, "Intern", "FastAPI, Kubernetes")

    assert len(report.bullets) == 1
    assert len(report.rejected) == 1
    assert report.rejected[0].claims_missing_skill == ["kubernetes"]


def test_stretched_bullet_stays_rejected_if_repair_fails(monkeypatch):
    fake = _tailored(TailoredBullet(text=STRETCH, source=GOOD))
    monkeypatch.setattr(tailor_pipeline, "tailor_resume", lambda *a, **k: fake)
    monkeypatch.setattr(tailor_pipeline, "judge_bullet", _judge_by_content)
    monkeypatch.setattr(
        tailor_pipeline, "rewrite_bullet", lambda *a, **k: "Built big data processing APIs."
    )

    report = tailor_pipeline.tailor_and_verify(PROFILE, "Intern", "FastAPI")

    assert report.bullets == []
    assert report.rejected[0].unsupported_claims == ["big data processing"]


def test_stretched_bullet_is_repaired(monkeypatch):
    fake = _tailored(TailoredBullet(text=STRETCH, source=GOOD))
    monkeypatch.setattr(tailor_pipeline, "tailor_resume", lambda *a, **k: fake)
    monkeypatch.setattr(tailor_pipeline, "judge_bullet", _judge_by_content)
    monkeypatch.setattr(tailor_pipeline, "rewrite_bullet", lambda *a, **k: GOOD + ".")

    report = tailor_pipeline.tailor_and_verify(PROFILE, "Intern", "FastAPI")

    assert report.rejected == []
    assert len(report.bullets) == 1
    assert report.bullets[0].repaired is True
    assert report.bullets[0].text == GOOD + "."