from app.services.llm_resume_parser import ResumeProfile, Project
from app.services.tailor import TailoredResume, TailoredBullet
from app.services.tailor_check import check_bullets

profile = ResumeProfile(
    name=None,
    email=None,
    skills=["python", "fastapi", "postgresql", "docker"],
    education=[],
    projects=[
        Project(
            name="Job Board API",
            description="Built a REST API with FastAPI and PostgreSQL",
            technologies=["fastapi", "postgresql"],
        )
    ],
    experience_summary=None,
)

result = TailoredResume(
    summary="Backend intern candidate.",
    bullets=[
        TailoredBullet(
            text="Built a REST API with FastAPI and PostgreSQL.",
            source="Built a REST API with FastAPI and PostgreSQL",
        ),
        TailoredBullet(
            text="Deployed services on Kubernetes at scale.",
            source="Led a Kubernetes migration at Google",
        ),
    ],
    matched_skills=["fastapi", "postgresql"],
    missing_skills=["kubernetes"],
)

for c in check_bullets(profile, result):
    print(c.ok, "|", c.text, "| grounded:", c.grounded, "| leaked:", c.claims_missing_skill)