from app.database import SessionLocal
from app.services.retrieval import hybrid_search

query = "python api developer internship"

with SessionLocal() as session:
    for mode in ("keyword", "vector", "hybrid"):
        print(mode)
        for job, score in hybrid_search(session, query, top_k=3, mode=mode):
            print("  ", job.title, round(score, 4))