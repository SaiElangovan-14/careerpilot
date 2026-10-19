import sys

from app.services.llm_resume_parser import parse_resume_llm
from app.services.resume_parser import extract_pdf_text, parse_resume_text

with open(sys.argv[1], "rb") as f:
    text = extract_pdf_text(f.read())

print("REGEX:", parse_resume_text(text))
print()
print("LLM:", parse_resume_llm(text).model_dump_json(indent=2))