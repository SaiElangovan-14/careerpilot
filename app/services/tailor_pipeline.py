from pydantic import BaseModel

from app.services.llm_resume_parser import ResumeProfile
from app.services.tailor import tailor_resume
from app.services.tailor_check import BulletCheck, check_bullets


class TailorReport(BaseModel):
    summary: str
    bullets: list[BulletCheck]    # passed verification
    rejected: list[BulletCheck]   # failed: invented source or claimed a missing skill
    matched_skills: list[str]
    missing_skills: list[str]


def tailor_and_verify(
    profile: ResumeProfile, job_title: str, job_description: str
) -> TailorReport:
    result = tailor_resume(profile, job_title, job_description)
    checks = check_bullets(profile, result)
    return TailorReport(
        summary=result.summary,
        bullets=[c for c in checks if c.ok],
        rejected=[c for c in checks if not c.ok],
        matched_skills=result.matched_skills,
        missing_skills=result.missing_skills,
    )