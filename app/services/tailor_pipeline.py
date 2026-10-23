from pydantic import BaseModel

from app.services.judge import judge_bullet
from app.services.llm_resume_parser import ResumeProfile
from app.services.repair import rewrite_bullet
from app.services.tailor import TailoredBullet, TailoredResume, tailor_resume
from app.services.tailor_check import BulletCheck, check_bullets


class TailorReport(BaseModel):
    summary: str
    bullets: list[BulletCheck]    # passed every check
    rejected: list[BulletCheck]   # failed at least one check
    matched_skills: list[str]
    missing_skills: list[str]


def _verify(
    profile: ResumeProfile, source: str, text: str, missing_skills: list[str]
) -> BulletCheck:
    single = TailoredResume(
        summary="",
        bullets=[TailoredBullet(text=text, source=source)],
        matched_skills=[],
        missing_skills=missing_skills,
    )
    check = check_bullets(profile, single)[0]
    if check.ok:
        verdict = judge_bullet(source, text)
        if not verdict.supported:
            check = check.model_copy(
                update={"ok": False, "unsupported_claims": verdict.unsupported_claims}
            )
    return check


def tailor_and_verify(
    profile: ResumeProfile, job_title: str, job_description: str
) -> TailorReport:
    result = tailor_resume(profile, job_title, job_description)

    kept: list[BulletCheck] = []
    rejected: list[BulletCheck] = []
    for bullet in result.bullets:
        check = _verify(profile, bullet.source, bullet.text, result.missing_skills)

        if not check.ok and check.grounded:
            problems = (
                check.unsupported_claims
                + check.claims_missing_skill
                + check.unsupported_numbers
            )
            new_text = rewrite_bullet(check.source, check.text, problems, job_title)
            retry = _verify(profile, check.source, new_text, result.missing_skills)
            if retry.ok:
                check = retry.model_copy(update={"repaired": True})

        (kept if check.ok else rejected).append(check)

    return TailorReport(
        summary=result.summary,
        bullets=kept,
        rejected=rejected,
        matched_skills=result.matched_skills,
        missing_skills=result.missing_skills,
    )