from types import SimpleNamespace

from app.services import rag_context


def make_chunk(chunk_id, text):
    return SimpleNamespace(chunk_id=chunk_id, text=text)


def test_format_context_labels_each_chunk_with_its_id():
    chunks = [
        make_chunk("project-0-1", "TenantCore: Implemented JWT authentication"),
        make_chunk("project-2-0", "CareerPilot: Built hybrid retrieval"),
    ]
    assert rag_context.format_context(chunks) == (
        "[project-0-1] TenantCore: Implemented JWT authentication\n"
        "[project-2-0] CareerPilot: Built hybrid retrieval"
    )


def test_format_context_empty():
    assert rag_context.format_context([]) == ""


def test_retrieve_context_filters_kinds_and_drops_scores(monkeypatch):
    seen = {}
    chunk = make_chunk("project-0-0", "TenantCore: Built a backend")

    def fake_search(session, resume_id, query, top_k, mode, kinds):
        seen.update(
            resume_id=resume_id, query=query, top_k=top_k, mode=mode, kinds=kinds
        )
        return [(chunk, 0.033)]

    monkeypatch.setattr(rag_context, "hybrid_search_chunks", fake_search)
    result = rag_context.retrieve_context(None, "r1", "Backend Intern", "Build APIs")

    assert result == [chunk]
    assert seen["resume_id"] == "r1"
    assert seen["mode"] == "hybrid"
    assert seen["kinds"] == ("project", "experience")
    assert seen["query"].startswith("Backend Intern Build APIs")