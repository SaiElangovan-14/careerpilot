import logging

logger = logging.getLogger("careerpilot")

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI 
from fastapi import File, Form, HTTPException,Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session 

from app.database import Job, SessionLocal,init_db
from app.services.embeddings import embed 
from app.services.retrieval import hybrid_search
from app.services.resume_profile import build_profile
from app.services.resume_parser import extract_pdf_text, parse_resume_text
from app.services.tailor_pipeline import TailorReport, tailor_and_verify, tailor_and_verify_rag
from app.services.cover_letter import LetterReport, draft_and_verify_letter, frame_letter
from app.services.rag_pipeline import build_context
from app.services.fit import FitReport, analyze_fit

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    embed(["warm up the model"])
    yield

app = FastAPI(title="CareerPilot", lifespan=lifespan)

def get_db():
    with SessionLocal() as session:
        yield session

class JobIn(BaseModel):
    title: str = Field(min_length=2, max_length=300)
    company: str = ""
    description: str = Field(min_length=20)

class JobOut(BaseModel):
    id: int 
    title: str
    company: str 
    score: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/jobs", status_code=201)
def add_jobs(jobs:list[JobIn], db: Session = Depends(get_db)):
    vectors = embed([f"{j.title} {j.description}" for j in jobs])
    for job,vec in zip(jobs,vectors):
        db.add(
            Job(
                title=job.title,
                company=job.company,
                description=job.description,
                embedding=vec,
            )
        )
    db.commit()
    return{"added": len(jobs)}

@app.get("/search", response_model=list[JobOut])
def search(
    q: str,
    top_k: int = 5,
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid",
    db: Session = Depends(get_db),
):
    results = hybrid_search(db, q, top_k=top_k, mode=mode)
    return [
        JobOut(id=job.id,title=job.title, company=job.company, score=score)
        for job, score in results
    ]

MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB


@app.post("/resumes")
def match_resume(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    data = file.file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF is larger than 5 MB")

    try:
        text = extract_pdf_text(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this PDF")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text found in PDF (is it a scanned image?)")

    profile, parser = build_profile(text)
    query = " ".join(profile.skills) or text[:500]
    results = hybrid_search(db, query, top_k=top_k)

    return {
        "parser": parser,
        "profile": profile,
        "matches": [
            JobOut(id=job.id, title=job.title, company=job.company, score=score)
            for job, score in results
        ],
    }

def _read_job_and_profile(db: Session, job_id: int, file: UploadFile):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF")

    data = file.file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF is larger than 5 MB")

    try:
        text = extract_pdf_text(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this PDF")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text found in PDF")

    profile, _parser = build_profile(text)
    return job, profile

@app.post("/tailor", response_model=TailorReport)
def tailor(
    job_id: int = Form(...),
    use_rag: bool = Form(True),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    job, profile = _read_job_and_profile(db, job_id, file)

    try:
        if use_rag:
            chunks = build_context(db, profile, job)
            return tailor_and_verify_rag(profile, chunks, job.title, job.description)
        return tailor_and_verify(profile, job.title, job.description)
    except RuntimeError as e:
        logger.warning("LLM call failed: %s", e)
        raise HTTPException(status_code=503, detail="The AI service is busy, try again shortly")


@app.post("/cover-letter", response_model=LetterReport)
def cover_letter_endpoint(
    job_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    job, profile = _read_job_and_profile(db, job_id, file)

    try:
        report = draft_and_verify_letter(profile, job.title, job.description)
        report.letter = frame_letter(report.body, job.title, job.company, profile.name)
        return report
    except RuntimeError as e:
        logger.warning("LLM call failed: %s", e)
        raise HTTPException(status_code=503, detail="The AI service is busy, try again shortly")

@app.post("/fit", response_model=FitReport)
def fit_endpoint(
    job_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    job, profile = _read_job_and_profile(db, job_id, file)

    try:
        chunks = build_context(db, profile, job, top_k=8)
        return analyze_fit(chunks, profile.skills, job.title, job.description)
    except RuntimeError as e:
        logger.warning("LLM call failed: %s", e)
        raise HTTPException(status_code=503, detail="The AI service is busy, try again shortly")