import pytest
from pydantic import BaseModel

from app.services import llm


class Out(BaseModel):
    value: str


class FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


def setup(monkeypatch, outcomes):
    client = FakeClient(outcomes)
    sleeps = []
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    return client, sleeps


def test_retries_after_overload_then_succeeds(monkeypatch):
    overload = Exception("503 UNAVAILABLE high demand")
    client, sleeps = setup(monkeypatch, [overload, overload, overload, Out(value="ok")])
    result = llm._generate_uncached("p", Out)
    assert result.value == "ok"
    assert client.calls == llm.MODELS + [llm.MODELS[0]]
    assert sleeps == [5]


def test_quota_exhausted_models_are_not_retried(monkeypatch):
    quota = Exception("429 RESOURCE_EXHAUSTED")
    client, sleeps = setup(monkeypatch, [quota, quota, quota])
    with pytest.raises(RuntimeError):
        llm._generate_uncached("p", Out)
    assert client.calls == llm.MODELS
    assert sleeps == []


def test_gives_up_after_all_rounds(monkeypatch):
    overload = Exception("503 UNAVAILABLE")
    client, sleeps = setup(monkeypatch, [overload] * 9)
    with pytest.raises(RuntimeError):
        llm._generate_uncached("p", Out)
    assert len(client.calls) == 9
    assert sleeps == [5, 15]

def test_model_that_times_out_is_skipped_for_the_rest_of_the_request(monkeypatch):
    from types import SimpleNamespace

    from app.services import llm

    calls = []

    def fake_generate_content(model, contents, config):
        calls.append(model)
        if model == llm.MODELS[0]:
            raise RuntimeError("504 DEADLINE_EXCEEDED")
        return SimpleNamespace(parsed="ok")

    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content))
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)

    result = llm._generate_uncached("prompt", object)

    assert result == "ok"
    assert calls == [llm.MODELS[0], llm.MODELS[1]]