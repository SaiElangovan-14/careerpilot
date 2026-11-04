import re

from pydantic import BaseModel

from app.services.llm_resume_parser import ResumeProfile


class Chunk(BaseModel):
    chunk_id: str
    kind: str
    title: str
    text: str


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -•\t") for p in parts if p.strip(" -•\t")]


def chunk_profile(profile: ResumeProfile) -> list[Chunk]:
    chunks: list[Chunk] = []

    for i, p in enumerate(profile.projects):
        for j, sentence in enumerate(_sentences(p.description)):
            chunks.append(
                Chunk(
                    chunk_id=f"project-{i}-{j}",
                    kind="project",
                    title=p.name,
                    text=f"{p.name}: {sentence}",
                )
            )
        if p.technologies:
            chunks.append(
                Chunk(
                    chunk_id=f"project-{i}-tech",
                    kind="project",
                    title=p.name,
                    text=f"{p.name} technologies: {', '.join(p.technologies)}",
                )
            )

    if profile.skills:
        chunks.append(
            Chunk(
                chunk_id="skills",
                kind="skills",
                title="Skills",
                text="Skills: " + ", ".join(profile.skills),
            )
        )

    for i, e in enumerate(profile.education):
        year = f" ({e.year})" if e.year else ""
        chunks.append(
            Chunk(
                chunk_id=f"education-{i}",
                kind="education",
                title=e.institution,
                text=f"{e.degree}, {e.institution}{year}",
            )
        )

    if profile.experience_summary:
        chunks.append(
            Chunk(
                chunk_id="experience",
                kind="experience",
                title="Experience",
                text=profile.experience_summary,
            )
        )

    return chunks