from types import SimpleNamespace

from app.services import fit


def chunk(chunk_id, text):
    return SimpleNamespace(chunk_id=chunk_id, text=text)


CHUNKS = [
    chunk("project-0-1", "TenantCore: Built RESTful APIs with SQLAlchemy."),
    chunk("project-1-1", "Job Board API: Built JWT authentication."),
]


def test_evidence_ids_are_mapped_to_resume_text(monkeypatch):
    model = fit.FitModel(
        requirements=[
            fit.RequirementFit(
                requirement="REST APIs",
                status="supported",
                evidence_ids=["project-0-1"],
                note="Built REST APIs.",
            )
        ]
    )
    monkeypatch.setattr(fit, "generate_structured", lambda prompt, schema: model)

    report = fit.analyze_fit(CHUNKS, ["Python"], "Backend Intern", "Needs REST APIs")

    assert report.items[0].evidence == ["TenantCore: Built RESTful APIs with SQLAlchemy."]
    assert report.supported == 1


def test_claim_without_valid_evidence_becomes_a_gap(monkeypatch):
    model = fit.FitModel(
        requirements=[
            fit.RequirementFit(
                requirement="Kubernetes",
                status="supported",
                evidence_ids=["does-not-exist"],
                note="Made up.",
            ),
            fit.RequirementFit(
                requirement="Docker",
                status="partial",
                evidence_ids=[],
                note="No evidence.",
            ),
        ]
    )
    monkeypatch.setattr(fit, "generate_structured", lambda prompt, schema: model)

    report = fit.analyze_fit(CHUNKS, [], "Backend Intern", "Needs Kubernetes")

    assert [i.status for i in report.items] == ["gap", "gap"]
    assert report.gaps == 2
    assert report.supported == 0