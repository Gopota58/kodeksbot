"""Герметичная конфигурация тестов.

Тест-сьют не должен зависеть ни от внешних моделей, ни от рабочего корпуса,
ни от `.env` разработчика. Поэтому до импорта `config` (pydantic-settings читает
os.environ при создании Settings) фиксируем окружение:

- EMBED_PROVIDER=hash  -> детерминированный хэш-эмбеддер (см. rag.engine), без torch;
- CHROMA_DIR=<temp>    -> временный каталог Chroma, чтобы тесты НЕ трогали рабочий chroma_db;
- DOCS_DIR=<fixtures>  -> маленький учебный корпус вместо 26 кодексов РФ
                          (data/docs/ в .gitignore, в CI его нет);
- RERANK_MODEL=""      -> reranker выключен: в CI моделей нет, а локально их загрузка
                          лишь замедлила бы тесты;
- API_KEY / ADMIN_API_KEY -> фиксированные значения, чтобы тесты не зависели от .env
                          и могли проверять разделение публичного и админского ключей.

Эндпоинт LLM в CI недоступен -> интеграционный /ask сам пропускается (см. tests/test_api.py),
а сквозной ask() покрыт моком (tests/test_ask_mock.py).
"""
import os
import pathlib
import tempfile

FIXTURE_DOCS = pathlib.Path(__file__).resolve().parent / "fixtures" / "docs"

os.environ["EMBED_PROVIDER"] = "hash"
os.environ["CHROMA_DIR"] = tempfile.mkdtemp(prefix="kodeksbot_test_")
os.environ["DOCS_DIR"] = str(FIXTURE_DOCS)
os.environ["RERANK_MODEL"] = ""
# Фоновый watcher (watchfiles) в тестах не нужен, а на Linux его поток падает
# при завершении процесса (SIGABRT) -> отключаем, иначе CI «краснеет» после успеха.
os.environ["RAG_DISABLE_WATCHER"] = "1"
os.environ["API_KEY"] = "test-public-key"
os.environ["ADMIN_API_KEY"] = "test-admin-key"
