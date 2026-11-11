from types import SimpleNamespace

from app.services import cover_letter


def chunk(chunk_id, text):
    return SimpleNamespace(chunk_id=chunk_id, text=text)


CHUNKS = [
    chunk("project-0-1", "TenantCore: Built RESTful APIs with SQLAlchemy."),
    chunk("project-1-1", "Job Board API: Built JWT authentication."),
]


def test_cited_ids_become_resume_text_as_source(monkeypatch):
    result = cover_letter.RagLetterBody(
        sentences=[
            cover_letter.RagLetterSentence(text="I built REST APIs.", chunk_id="project-0-1"),
            cover_letter.RagLetterSentence(text="I made up Kubernetes.", chunk_id="nope"),
        ]
    )
    monkeypatch.setattr(cover_letter, "generate_structured", lambda prompt, schema: result)

    body = cover_letter.draft_cover_letter_body_rag(CHUNKS, "Backend Intern", "Needs APIs")

    assert len(body.sentences) == 1  # the unknown id was dropped
    assert body.sentences[0].text == "I built REST APIs."
    assert body.sentences[0].source == "TenantCore: Built RESTful APIs with SQLAlchemy."


def test_no_chunks_gives_empty_report_without_calling_the_model(monkeypatch):
    def boom(prompt, schema):
        raise AssertionError("the model should not be called")

    monkeypatch.setattr(cover_letter, "generate_structured", boom)

    report = cover_letter.draft_and_verify_letter_rag(None, [], "Backend Intern", "Needs APIs")

    assert report.body == ""
    assert report.kept == [] and report.rejected == []