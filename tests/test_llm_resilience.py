import pytest
from pydantic import BaseModel

from app.services import llm


class Out(BaseModel):
    x: int


class FakeError(Exception):
    def __init__(self, code, text):
        super().__init__(f"{code} {text}")
        self.code = code


class FakeModels:
    def __init__(self, behavior):
        self.behavior = behavior  # model name -> exception to raise, or a value to return
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        result = self.behavior[model]
        if isinstance(result, Exception):
            raise result

        class Response:
            parsed = result

        return Response()


class FakeClient:
    def __init__(self, behavior):
        self.models = FakeModels(behavior)


def use_fake(monkeypatch, behavior):
    client = FakeClient(behavior)
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    return client


def test_auth_error_fails_fast(monkeypatch):
    err = FakeError(401, "UNAUTHENTICATED API key not valid")
    client = use_fake(monkeypatch, {m: err for m in llm.MODELS})
    with pytest.raises(llm.LLMAuthError):
        llm._generate_uncached("hi", Out)
    assert client.models.calls == [llm.MODELS[0]]


def test_deadline_stops_further_attempts(monkeypatch):
    err = FakeError(504, "DEADLINE_EXCEEDED")
    client = use_fake(monkeypatch, {m: err for m in llm.MODELS})
    monkeypatch.setattr(llm, "DEADLINE_SECONDS", 0)
    with pytest.raises(RuntimeError):
        llm._generate_uncached("hi", Out)
    assert client.models.calls == [llm.MODELS[0]]


def test_falls_back_to_next_model(monkeypatch):
    behavior = {
        llm.MODELS[0]: FakeError(503, "UNAVAILABLE"),
        llm.MODELS[1]: Out(x=1),
        llm.MODELS[2]: Out(x=2),
    }
    client = use_fake(monkeypatch, behavior)
    result = llm._generate_uncached("hi", Out)
    assert result == Out(x=1)
    assert client.models.calls == [llm.MODELS[0], llm.MODELS[1]]