from app.services.judge import judge_bullet

cases = [
    (
        "Job Board API project: Implemented job posting CRUD, search/filter/pagination, and an application workflow with status tracking.",
        "Implemented backend CRUD operations, search, filtering, and pagination for data processing in PostgreSQL.",
    ),
    (
        "Job Board API project: Implemented job posting CRUD, search/filter/pagination, and an application workflow with status tracking.",
        "Built job posting CRUD, search, filtering and pagination.",
    ),
]

for source, bullet in cases:
    v = judge_bullet(source, bullet)
    print(v.supported, "|", v.unsupported_claims, "|", bullet[:60])