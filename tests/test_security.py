import pytest
from fastapi import HTTPException

from app.security import require_api_key


def test_unconfigured_server_refuses_everything(monkeypatch):
    monkeypatch.delenv("CAREERPILOT_API_KEY", raising=False)
    with pytest.raises(HTTPException) as e:
        require_api_key("anything")
    assert e.value.status_code == 503


def test_missing_or_wrong_key_is_rejected(monkeypatch):
    monkeypatch.setenv("CAREERPILOT_API_KEY", "secret")
    for bad in (None, "", "wrong"):
        with pytest.raises(HTTPException) as e:
            require_api_key(bad)
        assert e.value.status_code == 401


def test_correct_key_passes(monkeypatch):
    monkeypatch.setenv("CAREERPILOT_API_KEY", "secret")
    assert require_api_key("secret") is None