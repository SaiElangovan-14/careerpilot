from app.services.llm_resume_parser import parse_resume_llm
from app.services.resume_parser import extract_pdf_text
from app.services.tailor_pipeline import tailor_and_verify

with open(r"C:\Users\saiel\Downloads\Sai_Elangovan_Resume-1.pdf", "rb") as f:
    text = extract_pdf_text(f.read())

profile = parse_resume_llm(text)

job = """Backend Developer Intern. Build REST APIs with FastAPI and PostgreSQL,
write tests, use Docker. Nice to have: Kubernetes, AWS, Redis."""

report = tailor_and_verify(profile, "Backend Developer Intern", job)

print("SUMMARY:", report.summary)
print("MATCHED:", report.matched_skills)
print("MISSING:", report.missing_skills)
print("\nKEPT:")
for b in report.bullets:
    print(" -", b.text)
print("\nREJECTED:")
for b in report.rejected:
    print(" -", b.text, "| grounded:", b.grounded, "| leaked:", b.claims_missing_skill)