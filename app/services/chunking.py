from pydantic import BaseModel

from app.services.llm_resume_parser import ResumeProfile


class Chunk(BaseModel):
    chunk_id: str
    kind: str
    title: str
    text: str


def chunk_profile(profile: ResumeProfile) -> list[Chunk]:
    chunks: list[Chunk] = []

    for i, p in enumerate(profile.projects):
        text = f"{p.name}: {p.description}"
        if p.technologies:
            text += f" Technologies: {', '.join(p.technologies)}."
        chunks.append(
            Chunk(chunk_id=f"project-{i}", kind="project", title=p.name, text=text)
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