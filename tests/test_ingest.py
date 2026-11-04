from app.services import ingest
from app.services.chunking import Chunk


class FakeSession:
    def __init__(self):
        self.added = []
        self.committed = False

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True


def make_chunks():
    return [
        Chunk(chunk_id="project-0", kind="project", title="TenantCore", text="TenantCore: SaaS"),
        Chunk(chunk_id="skills", kind="skills", title="Skills", text="Skills: Python"),
    ]


def test_ingest_saves_each_chunk(monkeypatch):
    monkeypatch.setattr(ingest, "has_resume", lambda s, r: False)
    monkeypatch.setattr(ingest, "embed", lambda texts: [[0.0] * 384 for _ in texts])
    session = FakeSession()
    count = ingest.ingest_chunks(session, "abc", make_chunks())
    assert count == 2
    assert session.committed
    assert [o.chunk_id for o in session.added] == ["project-0", "skills"]
    assert all(o.resume_id == "abc" for o in session.added)


def test_ingest_skips_existing_resume(monkeypatch):
    monkeypatch.setattr(ingest, "has_resume", lambda s, r: True)
    session = FakeSession()
    assert ingest.ingest_chunks(session, "abc", make_chunks()) == 0
    assert session.added == []


def test_resume_id_is_stable():
    assert ingest.resume_id_for("hello") == ingest.resume_id_for("hello")
    assert ingest.resume_id_for("hello") != ingest.resume_id_for("hello!")