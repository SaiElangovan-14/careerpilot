from app.services.llm_resume_parser import Project, ResumeProfile
from app.services.tailor import TailoredBullet, TailoredResume
from app.services.tailor_check import check_bullets

PROFILE = ResumeProfile(
    name=None,
    email=None,
    skills=["python", "fastapi", "postgresql", "docker"],
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


def _result(bullets, missing=None):
    return TailoredResume(
        summary="x",
        bullets=bullets,
        matched_skills=[],
        missing_skills=missing or [],
    )


def test_grounded_bullet_passes():
    result = _result(
        [TailoredBullet(
            text="Built a REST API with FastAPI and PostgreSQL.",
            source="Built a REST API with FastAPI and PostgreSQL",
        )]
    )
    check = check_bullets(PROFILE, result)[0]
    assert check.ok


def test_invented_source_is_flagged():
    result = _result(
        [TailoredBullet(
            text="Led a team of ten engineers.",
            source="Led a team of ten engineers at Google",
        )]
    )
    check = check_bullets(PROFILE, result)[0]
    assert not check.grounded
    assert not check.ok


def test_missing_skill_in_bullet_is_flagged():
    result = _result(
        [TailoredBullet(
            text="Deployed Docker containers on Kubernetes.",
            source="Docker",
        )],
        missing=["kubernetes"],
    )
    check = check_bullets(PROFILE, result)[0]
    assert check.grounded
    assert check.claims_missing_skill == ["kubernetes"]
    assert not check.ok

def test_invented_number_is_flagged():
    result = _result(
    [TailoredBullet(
        text="Built a REST API with FastAPI serving 5000 users.",
        source="Built a REST API with FastAPI and PostgreSQL",
    )]
    )
    check = check_bullets(PROFILE, result)[0]
    assert check.grounded
    assert check.unsupported_numbers == ["5000"]
    assert not check.ok

        