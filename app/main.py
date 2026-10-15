from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI 
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session 

from app.database import Job, SessionLocal,init_db
from app.services.embeddings import embed 
from app.services.retrieval import hybrid_search

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