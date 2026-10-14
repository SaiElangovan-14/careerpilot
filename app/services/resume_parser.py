import re 

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

SKILLS = {"python", "fastapi", "sql", "postgresql", "docker","redis","git","java","javascript","c++"}

def parse_resume_text(text: str) -> dict:
    lowered = text.lower()
    skills = sorted(
        s for s in SKILLS 
        if re.search(rf"(?<![\w+]){re.escape(s)}(?![\w+])", lowered)
    )
    match = EMAIL_RE.search(text)
    return {"email": match.group(0) if match else None, "skills": skills}