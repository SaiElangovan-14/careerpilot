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

        checks.append(
            BulletCheck(
                text=bullet.text,
                source=bullet.source,
                grounded=grounded,
                claims_missing_skill=leaked,
                ok=grounded and not leaked,
            )
        )
    return checks