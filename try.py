from app.services.repair import rewrite_bullet

source = (
    "Built a dual-role job platform backend supporting Employer and Candidate accounts "
    "with JWT authentication and ownership-based access control."
)
bullet = "Designed a Job Board API supporting search, pagination, CRUD operations, and role-based access control."

print(rewrite_bullet(source, bullet, ["role-based access control"], "Data Engineer Intern"))