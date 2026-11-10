import sys

from sqlalchemy import select

from app.database import Job, SessionLocal
from app.main import build_profile, extract_pdf_text
from app.services.rag_context import format_context
from app.services.rag_pipeline import build_context

with open(sys.argv[1], "rb") as f:
    text = extract_pdf_text(f.read())

profile, _parser = build_profile(text)

with SessionLocal() as session:
    for job in session.scalars(select(Job).limit(3)):
        print("\nJOB:", job.title)
        print(format_context(build_context(session, profile, job)))