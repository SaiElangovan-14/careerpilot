# careerpilot
## Evaluation

Golden set: 30 hand-written job postings and 12 queries with labeled relevant jobs
(`evals/data/golden.json`). Run it against a throwaway database:

    docker compose exec db psql -U careerpilot -d careerpilot -c "CREATE DATABASE careerpilot_eval;"
    $env:DATABASE_URL="postgresql+psycopg2://careerpilot:careerpilot@localhost:5433/careerpilot_eval"
    python -m evals.run_eval evals/data/golden.json

| Date | Change | Mode | recall@5 | hit@5 | MRR |
|---|---|---|---|---|---|
| Oct 16 | baseline (bge-small-en-v1.5) | keyword | 0.812 | 1.000 | 1.000 |
| Oct 16 | baseline (bge-small-en-v1.5) | vector | 0.917 | 1.000 | 0.958 |
| Oct 16 | baseline (bge-small-en-v1.5) | hybrid (RRF) | 0.917 | 1.000 | 1.000 |
| Oct 16 | hybrid + cross-encoder rerank (ms-marco-MiniLM-L-6-v2) | rerank | 0.889 | 1.000 | 0.944 |

Hybrid matches the best single method on every metric. Off-the-shelf reranking did not beat
plain hybrid on this set (trained on web Q&A, not job search), so hybrid stays the default.
The dataset is small and hit@5 is saturated, so a larger, harder set is planned. Next:
fine-tune on resume-job pairs (Phase 3).

## Resume parsing: regex vs LLM

One resume, 16 hand-labeled skills (`evals/data/resume_gold.json`). Run:

    python -m evals.parser_eval <resume.pdf> evals/data/resume_gold.json

| Parser | Skills found | Precision | Recall |
|---|---|---|---|
| Regex (fixed skill list) | 6 | 1.000 | 0.375 |
| LLM (Gemini, Pydantic-validated) | 16 | 1.000 | 1.000 |

The regex parser can only find skills in a hard-coded list, so it misses anything outside it
(SQLAlchemy, Pydantic, Pytest, JWT, bcrypt, Cloudinary and more). The LLM parser returns a
validated `ResumeProfile` (skills, education, projects) and is told to use only skills named
in the resume. Caveat: one resume, so this is a smoke test. More labeled resumes are planned.