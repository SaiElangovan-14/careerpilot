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

## What CareerPilot does

CareerPilot is an AI copilot for job applications. Given a job posting and a resume, it:

1. **Searches jobs** with hybrid retrieval: Postgres full-text search plus pgvector embeddings, merged with Reciprocal Rank Fusion.
2. **Parses resumes** into a structured profile using an LLM with a fixed schema.
3. **Tailors resume bullets** to a job, then checks every bullet before it is shown:
   - each bullet must cite a source fact from the profile;
   - a rule-based check catches invented sources, invented numbers, and skills the candidate does not have;
   - an LLM judge checks that the bullet claims nothing beyond its source;
   - a rejected bullet gets one rewrite attempt and is re-checked, otherwise it is dropped.

## How quality is measured

Retrieval, resume parsing, and the faithfulness judge each have a labelled eval set under `evals/`:

### Faithfulness judge

An LLM judge checks every generated resume bullet against the source fact it cites. I measured it on a 24-case labelled set (10 faithful rewordings, 14 bullets with an inserted fault such as an invented number, a new tool, a bigger scale, a skill-level word, or a blend of two projects).

| Metric | Result |
|---|---|
| Unfaithful bullets caught | 14 / 14 |
| Faithful bullets wrongly flagged | 0 / 10 |

Caveats: the set is small, and the skill-level rule was added after the judge missed one case, so this is a regression test and not a general accuracy claim. Next: a larger set built from real job postings.

## Retrieval-augmented generation (RAG)

CareerPilot grounds every generated resume bullet in retrieved facts from the candidate's own resume.

**Pipeline**
1. The parsed resume is split into sentence-level fact chunks (`project-0-1`, ...), each with a kind (`project`, `tech`, `skills`, `education`, `summary`).
2. Chunks are embedded with `bge-small-en-v1.5` and stored in Postgres with pgvector. Ingest is idempotent: the resume id is a hash of the profile plus the chunker version.
3. For a job, hybrid retrieval (Postgres full-text + vector search, fused with Reciprocal Rank Fusion) returns the most relevant project facts.
4. The generator sees only those facts, each labelled with its id, and must cite a chunk id per bullet. Unknown ids are dropped.
5. Every bullet is then verified against its cited chunk: token-overlap grounding, an unsupported-number check, a skill-claim check, and an LLM judge. Failing bullets get one repair attempt, then are rejected.

**Retrieval eval** (8 labelled jobs against a synthetic resume, recall@5 / MRR):

| mode | before `tech` chunk fix | after |
|---|---|---|
| keyword | 0.86 / 0.94 | 0.90 / 1.00 |
| vector | 0.89 / 0.92 | 0.96 / 1.00 |
| hybrid | 0.97 / 0.85 | **1.00 / 1.00** |

The fix: technology lists were crowding the real facts out of the top results, so they now have their own chunk kind and are excluded from bullet context. A cross-encoder reranker was also tried and did not beat hybrid, so it is not used.

**Caveats:** the eval set is small (8 jobs) and synthetic, and I changed the chunker after seeing it, so it is partly tuned to it. A larger set with real jobs is the next step. Run it with `python -m evals.retrieval_eval` (no API calls).

Real postings (5 internship postings, 1 real resume, 13 candidate chunks; labels written
before running; random baseline recall@5 ~ 0.38):

| mode | recall@5 | MRR |
|---|---|---|
| keyword | 0.74 | 1.00 |
| vector | 0.66 | 0.55 |
| hybrid | 0.64 | 0.90 |

Hybrid did not beat keyword on real postings, unlike on the synthetic set. With 5 jobs
the differences are about one chunk, so I treat this as inconclusive. Next step: more
postings and a second resume before changing the retriever.

Live check (1 job, 1 real resume, Gemini): both modes pass verification. RAG mode
attributes each bullet to a named project and produces a more specific summary; bullet
text is otherwise similar. With a resume of ~10 facts, retrieval is not selective, so
this is a sanity check, not a quality benchmark. Retrieval quality is measured
separately in evals/retrieval_eval.py (8 synthetic jobs; small, and the chunker was
tuned after seeing results).

### Fit and gaps (`POST /fit`)
For a job, lists the main requirements and marks each supported / partial / gap,
citing the resume sentences that back it up. Evidence ids returned by the model are
mapped to real resume text in code; a claim of support with no valid evidence is
downgraded to a gap, and a skill that appears only in the skills list is capped at
partial. Live check (2 jobs, 1 real resume): Backend Intern -> 4 supported, 2 partial,
0 gaps; Frontend Developer -> 0 supported, 0 partial, 5 gaps. Small sample, one resume.

Knowledge file: extra true facts per project (knowledge/projects.txt) are merged into the
parsed profile, so retrieval, citation and verification all see them. Live check (Backend
Intern, 1 resume): Python and FastAPI moved from partial to supported once the file listed
them as technologies. Technology lists count as weaker evidence than described work.

### Cover letter (`POST /cover-letter`)
Retrieves the most relevant resume facts, asks the model for sentences that each cite
one fact id, maps ids back to resume text, then runs the same verification as bullets
(grounding, numbers, skill claims, judge). Live check (1 job, 1 resume): RAG mode kept
4/4 sentences; classic mode kept 3/4 and rejected a sentence that claimed the project
"aligns with the job's requirements". Known limitation: because relevance claims are
rejected, letters read as a list of verified facts rather than persuasive prose.

### Security
- AI and write endpoints require an `X-API-Key` header (`CAREERPILOT_API_KEY` in `.env`).
  The server refuses to run open: with no key configured, protected routes return 503.
  Keys are compared in constant time.
- AI endpoints are rate limited (10 requests per minute per client, 429 with Retry-After).
  Known limit: the counters are in memory, so they reset on restart and are not shared
  across processes; a real deployment would use Redis.
- Resume and job text is treated as untrusted data in every prompt, and every generated
  sentence is verified against the resume before it is returned.
