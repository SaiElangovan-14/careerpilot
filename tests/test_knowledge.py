from pydantic import BaseModel

from app.services import knowledge


class P(BaseModel):
    name: str
    description: str
    technologies: list[str]


class Profile(BaseModel):
    projects: list[P]


def test_load_facts_skips_comments_and_bad_lines(tmp_path):
    f = tmp_path / "projects.txt"
    f.write_text("# note\n\nTenantCore | Used Docker Compose\nno separator\n", encoding="utf-8")

    assert knowledge.load_facts(f) == {"tenantcore": ["Used Docker Compose"]}


def test_missing_file_gives_no_facts(tmp_path):
    assert knowledge.load_facts(tmp_path / "nope.txt") == {}


def test_facts_are_merged_into_the_matching_project():
    profile = Profile(projects=[P(name="TenantCore", description="Built APIs.", technologies=["PostgreSQL"])])
    facts = {"tenantcore": ["Used Docker Compose", "tech: Python, postgresql"]}

    merged = knowledge.add_knowledge(profile, facts)

    project = merged.projects[0]
    assert project.description == "Built APIs. Used Docker Compose."
    assert project.technologies == ["PostgreSQL", "Python"]  # no duplicate
    assert profile.projects[0].description == "Built APIs."  # original untouched


def test_unknown_project_is_ignored():
    profile = Profile(projects=[P(name="TenantCore", description="Built APIs.", technologies=[])])

    merged = knowledge.add_knowledge(profile, {"other": ["Something"]})

    assert merged.projects[0].description == "Built APIs."