"""
Модульный тест провайдера gigachat БЕЗ реального ключа и сети.

Подменяем langchain_gigachat.GigaChat фиктивным объектом, чтобы проверить,
что build_llm() при LLM_PROVIDER=gigachat корректно пробрасывает в SDK:
ключ (credentials), модель, base_url и настройки SSL. Это даёт приёмку
«тест с mock-GigaChat» из роадмапа без обращения к API Сбера.
"""
from types import SimpleNamespace
from unittest.mock import patch

from rag.engine import build_llm


def _fake_settings(**overrides):
    # Минимальный объект настроек, имитирующий config.Settings для ветки gigachat.
    defaults = dict(
        llm_provider="gigachat",
        llm_api_key="test-auth-key",
        llm_model="GigaChat-2",
        llm_temperature=0.4,
        llm_top_p=0.9,
        llm_max_tokens=512,
        gigachat_base_url="https://api.giga.chat/v1",
        gigachat_verify_ssl_certs=True,
        gigachat_ca_bundle_file="",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_llm_gigachat_wires_credentials_and_model():
    settings = _fake_settings()
    with patch("langchain_gigachat.GigaChat") as mock_gc:
        build_llm(settings)
        # SDK должен быть вызван ровно один раз с нашими параметрами.
        assert mock_gc.called, "GigaChat из langchain-gigachat должен быть вызван"
        kwargs = mock_gc.call_args.kwargs
        # Ключ из LLM_API_KEY идёт в credentials; модель — обязательный параметр.
        assert kwargs["credentials"] == "test-auth-key"
        assert kwargs["model"] == "GigaChat-2"
        assert kwargs["base_url"] == "https://api.giga.chat/v1"
        assert kwargs["verify_ssl_certs"] is True
        assert "max_tokens" in kwargs and kwargs["max_tokens"] == 512


def test_build_llm_gigachat_passes_ca_bundle_when_set():
    settings = _fake_settings(gigachat_ca_bundle_file="C:/certs/ca.crt")
    with patch("langchain_gigachat.GigaChat") as mock_gc:
        build_llm(settings)
        assert mock_gc.call_args.kwargs.get("ca_bundle_file") == "C:/certs/ca.crt"


def test_build_llm_gigachat_requires_model():
    # Без LLM_MODEL провайдер gigachat обязан упасть с понятной ошибкой.
    settings = _fake_settings(llm_model="")
    with patch("langchain_gigachat.GigaChat"):
        try:
            build_llm(settings)
        except ValueError:
            return
        raise AssertionError("Ожидалась ValueError при пустом LLM_MODEL для gigachat")
