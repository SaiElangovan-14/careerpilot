import logging

from app.services.llm_resume_parser import ResumeProfile, parse_resume_llm
from app.services.resume_parser import parse_resume_text

logger = logging.getLogger(__name__)


def build_profile(text: str) -> tuple[ResumeProfile, str]:
    try:
        return parse_resume_llm(text), "llm"
    except Exception:
        logger.warning("LLM resume parsing failed, falling back to regex", exc_info=True)
        parsed = parse_resume_text(text)
        profile = ResumeProfile(
            name=None,
            email=parsed["email"],
            skills=parsed["skills"],
            education=[],
            projects=[],
            experience_summary=None,
        )
        return profile, "regex"