"""
Тесты HTTP-API через FastAPI TestClient (в процессе, без сети к uvicorn).

Покрывают: /health, аутентификацию, разделение публичного и админского ключей,
rate limiting (включая то, что CORS-preflight лимит не расходует) и список документов.
/ask интеграционный: выполняется, только если доступен LLM-эндпоинт,
иначе пропускается (актуально для CI без LLM).
"""
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app as fastapi_app
from config import settings

client = TestClient(fastapi_app)

# Публичный ключ открывает ТОЛЬКО /ask. Админский — всё, что меняет состояние.
PUBLIC_HEADERS = {"X-API-Key": settings.api_key}
ADMIN_HEADERS = {"X-API-Key": settings.resolved_admin_api_key}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_auth_required_for_ask():
    r = client.post("/ask", json={"question": "привет"})
    assert r.status_code == 401


def test_auth_ok_documents_list():
    r = client.get("/documents", headers=ADMIN_HEADERS)
    assert r.status_code == 200
    assert "documents" in r.json()
    # корпус подменён фикстурами (tests/fixtures/docs) — см. conftest.py
    assert len(r.json()["documents"]) > 0


# ============================================
# Разделение ключей: публичный ключ не пускает в админку
# ============================================

def test_keys_are_distinct():
    """Если ключи когда-нибудь сравняют, тесты ниже перестанут ловить регресс."""
    assert settings.api_key != settings.resolved_admin_api_key


def test_public_key_rejected_on_documents():
    r = client.get("/documents", headers=PUBLIC_HEADERS)
    assert r.status_code == 401


def test_public_key_rejected_on_ingest():
    r = client.post("/ingest", headers=PUBLIC_HEADERS)
    assert r.status_code == 401


def test_admin_endpoints_require_key():
    assert client.get("/documents").status_code == 401
    assert client.post("/ingest").status_code == 401


# ============================================
# Rate limiting
# ============================================

def _ask_with_wrong_key():
    """Запрос к /ask с заведомо неверным ключом: до LLM не доходит, квоту не тратит."""
    return client.post(
        "/ask",
        json={"question": "x"},
        headers={"X-API-Key": "definitely-wrong"},
    )


def test_rate_limit_returns_429(monkeypatch):
    app_module._rl_hits.clear()          # не зависим от соседних тестов
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)

    codes = [_ask_with_wrong_key().status_code for _ in range(3)]
    assert codes[:2] == [401, 401], "первые запросы должны дойти до проверки ключа"
    assert codes[2] == 429, "третий запрос обязан упереться в лимит"


def test_rate_limit_returns_retry_after(monkeypatch):
    app_module._rl_hits.clear()
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)

    _ask_with_wrong_key()
    r = _ask_with_wrong_key()
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert int(r.headers["Retry-After"]) >= 1


def test_preflight_options_not_rate_limited(monkeypatch):
    """CORS-preflight не должен расходовать лимит: он не доходит до LLM."""
    app_module._rl_hits.clear()
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)

    for _ in range(5):
        client.options("/ask", headers={"Origin": "http://example.com"})

    assert _ask_with_wrong_key().status_code == 401


def test_rate_limit_can_be_disabled(monkeypatch):
    app_module._rl_hits.clear()
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)

    codes = [_ask_with_wrong_key().status_code for _ in range(4)]
    assert codes == [401] * 4, "при rate_limit_enabled=False лимит не применяется"


# ============================================
# Интеграционный /ask (нужен доступный LLM)
# ============================================

def test_ask_integration():
    r = client.post(
        "/ask",
        json={"question": "Сколько раз в день кормить рыжего кота?"},
        headers=PUBLIC_HEADERS,
    )
    if r.status_code == 200:
        assert "answer" in r.json()
        assert isinstance(r.json()["answer"], str) and len(r.json()["answer"]) > 0
    else:
        pytest.skip("LLM endpoint unavailable (set LLM_BASE_URL to enable /ask test)")
