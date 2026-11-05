from types import SimpleNamespace

from app.services import tailor
from app.services.tailor import RagBullet, RagTailored


def make_chunks():
    return [
        SimpleNamespace(
            chunk_id="project-0-0", text="TenantCore: Built a multi-tenant SaaS backend"
        ),
        SimpleNamespace(
            chunk_id="project-1-1",
            text="Job Board API: Containerised the API with Docker Compose",
        ),
    ]


def fake_result():
    return RagTailored(
        summary="s",
        bullets=[
            RagBullet(text="Built a SaaS backend", chunk_id="project-0-0"),
            RagBullet(text="Invented thing", chunk_id="project-9-9"),
        ],
        matched_skills=["Python"],
        missing_skills=["Kubernetes"],
    )


def test_to_tailored_uses_chunk_text_as_source_and_drops_unknown_ids():
    result = tailor.to_tailored(fake_result(), make_chunks())
    assert [b.text for b in result.bullets] == ["Built a SaaS backend"]
    assert result.bullets[0].source == "TenantCore: Built a multi-tenant SaaS backend"
    assert result.missing_skills == ["Kubernetes"]


def test_tailor_resume_rag_prompt_has_ids_and_skills(monkeypatch):
    seen = {}

    def fake_generate(prompt, schema):
        seen["prompt"] = prompt
        seen["schema"] = schema
        return fake_result()

    monkeypatch.setattr(tailor, "generate_structured", fake_generate)
    result = tailor.tailor_resume_rag(
        make_chunks(), ["Python", "FastAPI"], "Backend Intern", "Build APIs"
    )
    assert "[project-0-0] TenantCore" in seen["prompt"]
    assert "Python, FastAPI" in seen["prompt"]
    assert seen["schema"] is RagTailored
    assert len(result.bullets) == 1