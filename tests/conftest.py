"""Герметичная конфигурация тестов.

Тест-сьют не должен зависеть от внешних моделей: ни от torch/HF-эмбеддингов,
ни от запущенного LLM/embedding-сервера. Поэтому до импорта `config`
(pydantic-settings читает os.environ при создании Settings) фиксируем:

- EMBED_PROVIDER=hash  -> детерминированный хэш-эмбеддер (см. rag.engine);
- CHROMA_DIR=<temp>    -> временный каталог Chroma, чтобы тесты НЕ трогали
                          рабочий chroma_db разработчика.

Эндпоинт LLM в CI недоступен -> интеграционный /ask сам пропускается (см.
tests/test_api.py), а сквозной ask() покрыт моком (tests/test_ask_mock.py).
"""
import os
import tempfile

os.environ["EMBED_PROVIDER"] = "hash"
os.environ["CHROMA_DIR"] = tempfile.mkdtemp(prefix="kotobaza_test_")
# Фоновый watcher (watchfiles) в тестах не нужен, а на Linux его поток падает
# при завершении процесса (SIGABRT) -> отключаем, иначе CI «краснеет» после успеха.
os.environ["RAG_DISABLE_WATCHER"] = "1"
