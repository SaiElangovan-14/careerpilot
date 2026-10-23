import re

from pydantic import BaseModel

from app.services.llm_resume_parser import ResumeProfile
from app.services.tailor import TailoredResume


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#]+", text.lower())


class BulletCheck(BaseModel):
    text: str
    source: str
    grounded: bool                    # is the cited source really in the resume?
    claims_missing_skill: list[str]   # skills the resume lacks but the bullet mentions
    unsupported_numbers: list[str]    # numbers in the bullet that are not in the resume
    unsupported_claims: list[str] =[]
    repaired: bool = False 
    ok: bool


def check_bullets(
    profile: ResumeProfile, result: TailoredResume, threshold: float = 0.8
) -> list[BulletCheck]:
    corpus = set(_tokens(profile.model_dump_json()))
    checks = []
    for bullet in result.bullets:
        source_tokens = _tokens(bullet.source)
        found = sum(t in corpus for t in source_tokens)
        grounded = bool(source_tokens) and found / len(source_tokens) >= threshold

        bullet_tokens = set(_tokens(bullet.text))
        leaked = [
            skill
            for skill in result.missing_skills
            if _tokens(skill) and set(_tokens(skill)) <= bullet_tokens
        ]

        numbers = re.findall(r"\d+", bullet.text)
        unsupported = [n for n in numbers if n not in corpus]

        checks.append(
            BulletCheck(
                text=bullet.text,
                source=bullet.source,
                grounded=grounded,
                claims_missing_skill=leaked,
                unsupported_numbers=unsupported,
                ok=grounded and not leaked and not unsupported,
            )
        )
    return checks

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