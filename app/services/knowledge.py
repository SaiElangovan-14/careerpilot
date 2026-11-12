import logging
from pathlib import Path

logger = logging.getLogger("careerpilot")

KNOWLEDGE_FILE = Path("knowledge/projects.txt")


def load_facts(path: Path = KNOWLEDGE_FILE) -> dict[str, list[str]]:
    """Read 'Project name | fact' lines. Lines starting with # are comments."""
    facts: dict[str, list[str]] = {}
    if not path.exists():
        return facts
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        name, fact = line.split("|", 1)
        name, fact = name.strip().lower(), fact.strip()
        if name and fact:
            facts.setdefault(name, []).append(fact)
    return facts


def add_knowledge(profile, facts: dict[str, list[str]]):
    """Return a copy of the profile with extra facts merged into its projects.

    A fact starting with 'tech:' adds technologies; any other fact becomes
    an extra sentence of the project description.
    """
    if not facts:
        return profile

    known = {p.name.strip().lower() for p in profile.projects}
    for name in facts.keys() - known:
        logger.warning("Knowledge file mentions unknown project %r; ignored", name)

    projects = []
    for project in profile.projects:
        extra = facts.get(project.name.strip().lower(), [])
        if extra:
            sentences = []
            technologies = list(project.technologies)
            for fact in extra:
                if fact.lower().startswith("tech:"):
                    for tech in fact[5:].split(","):
                        tech = tech.strip()
                        if tech and tech.lower() not in {t.lower() for t in technologies}:
                            technologies.append(tech)
                else:
                    sentences.append(fact if fact.endswith(".") else fact + ".")
            description = " ".join([project.description.strip(), *sentences]).strip()
            project = project.model_copy(
                update={"description": description, "technologies": technologies}
            )
        projects.append(project)

    return profile.model_copy(update={"projects": projects})