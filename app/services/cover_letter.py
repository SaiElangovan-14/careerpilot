import re

from app.services.llm_resume_parser import Project, ResumeProfile

STOPWORDS = {
    "the", "and", "for", "with", "you", "our", "are", "will", "your", "that",
    "this", "from", "have", "has", "not", "but", "all", "can", "use", "using",
    "work", "team", "role", "job", "who", "what", "into", "about", "more",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9+#]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def score_project(project: Project, job_tokens: set[str]) -> int:
    text_tokens = _tokens(project.name + " " + project.description)
    tech_tokens = _tokens(" ".join(project.technologies))
    return len(text_tokens & job_tokens) + 2 * len(tech_tokens & job_tokens)


def select_projects(
    profile: ResumeProfile, job_title: str, job_description: str, k: int = 2
) -> list[Project]:
    job_tokens = _tokens(job_title + " " + job_description)
    scored = [
        (score_project(p, job_tokens), i, p) for i, p in enumerate(profile.projects)
    ]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [p for _, _, p in scored[:k]]