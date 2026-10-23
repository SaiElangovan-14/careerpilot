from pydantic import BaseModel

from app.services.judge import judge_bullet
from app.services.llm_resume_parser import ResumeProfile
from app.services.tailor import tailor_resume
from app.services.tailor_check import BulletCheck, check_bullets


class TailorReport(BaseModel):
    summary: str
    bullets: list[BulletCheck]    # passed every check
    rejected: list[BulletCheck]   # failed at least one check
    matched_skills: list[str]
    missing_skills: list[str]


def tailor_and_verify(
    profile: ResumeProfile, job_title: str, job_description: str
) -> TailorReport:
    result = tailor_resume(profile, job_title, job_description)

    kept: list[BulletCheck] = []
    rejected: list[BulletCheck] = []
    for check in check_bullets(profile, result):
        if check.ok:
            verdict = judge_bullet(check.source, check.text)
            if not verdict.supported:
                check = check.model_copy(
                    update={"ok": False, "unsupported_claims": verdict.unsupported_claims}
                )
        (kept if check.ok else rejected).append(check)

    return TailorReport(
        summary=result.summary,
        bullets=kept,
        rejected=rejected,
        matched_skills=result.matched_skills,
        missing_skills=result.missing_skills,
    )