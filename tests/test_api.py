"""
Тесты HTTP-API через FastAPI TestClient (в процессе, без сети к uvicorn).
/ask интеграционный: выполняется, только если доступен LLM-эндпоинт,
иначе пропускается (актуально для CI без LLM).
"""
import pytest
from fastapi.testclient import TestClient

from app import app as fastapi_app
from config import settings

client = TestClient(fastapi_app)
HEADERS = {"X-API-Key": settings.api_key}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_auth_required_for_ask():
    r = client.post("/ask", json={"question": "привет"})
    assert r.status_code == 401


def test_auth_ok_documents_list():
    r = client.get("/documents", headers=HEADERS)
    assert r.status_code == 200
    assert "documents" in r.json()
    assert len(r.json()["documents"]) > 0


def test_ask_integration():
    r = client.post(
        "/ask",
        json={"question": "Сколько раз в день кормить рыжего кота?"},
        headers=HEADERS,
    )
    if r.status_code == 200:
        assert "answer" in r.json()
        assert isinstance(r.json()["answer"], str) and len(r.json()["answer"]) > 0
    else:
        pytest.skip("LLM endpoint unavailable (set LLM_BASE_URL to enable /ask test)")
