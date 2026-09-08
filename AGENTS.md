# AGENTS.md — КодексБот (юридический RAG-ассистент)

## 1. Суть проекта
RAG-ассистент для юридических документов: отвечает на вопросы по пользовательским
файлам (PDF/DOCX/TXT) с обязательным цитированием источника. Стек: Python +
FastAPI + LangChain + Chroma (векторная БД) + GigaChat (Сбер, через langchain-gigachat)
+ локальные эмбеддинги sentence-transformers для генерации. Статус: каркас собран на базе фреймворка «Котобаза» (FastAPI+LangChain+Chroma); корпус из 26 кодексов РФ загружен в data/docs/.

## 2. Карта папок
- `rag/engine.py` — ядро RAG: эмбеддинги, Chroma, гибридный ретривер (вектор+BM25→RRF), GigaChat, reindex, watcher
- `app.py` — FastAPI: `/ask` (+источники), `/ingest`, `/upload`, `/documents`, `/health`, CORS, статикa
- `config.py` — настройки из `.env` (pydantic-settings): LLM / эмбеддинги / Chroma
- `ingest.py` — CLI: `python ingest.py` (строит индекс Chroma из data/docs/)
- `static/` — веб-чат (vanilla JS); красивый UI обсуждается отдельно
- `bot.py` / `desktop_client.py` — Telegram-бот / десктоп (опц., из фреймворка)
- `data/docs/` — корпус: 26 кодексов РФ (ГК×4, НК×2, ТК, УК, процессуальные, отраслевые, КоАП, Конституция). Состав — `data/docs/README.md`.
- `chroma_db/` — векторная БД Chroma (persistent)
- `docs/` — подробная документация (Уровень 2)
- `requirements.txt`, `.env.example`, `Dockerfile`, `docker-compose.yml` — манифесты/деплой

## 3. Команды
- Установка зависимостей: `pip install -r requirements.txt`
- Индексация: `python ingest.py` (читает `data/docs/`, пишет `chroma_db/`)
- Запуск локально: `uvicorn app:app --reload --port 8000`
- Открыть UI: http://localhost:8000
- Проверка чата: `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -H "X-API-Key: 88888888" -d "{\"question\":\"...\"}"`
- Деплой: см. docs/DEPLOY.md (Yandex Cloud)

## 4. Грабли и правила
- GigaChat подключён «родным» `langchain-gigachat`: в `.env` → `LLM_PROVIDER=gigachat`, `LLM_API_KEY` = Authorization key из developers.sber.ru, `LLM_MODEL` = id из `GET /v1/models`. Ключ — только в `.env`, не коммитить.
- На Windows для GigaChat нужен российский root-CA: `GIGACHAT_CA_BUNDLE_FILE` (gosuslugi.ru/crt) либо `GIGACHAT_VERIFY_SSL_CERTS=false` для dev.
- Эмбеддинги локальные: `cointegrated/rubert-tiny2` (demo) или `ai-forever/sbert_large_nlu_ru` (точность). Модель лежит в `models/rubert-tiny2` и грузится **строго локально, без сети** (`HF_HUB_OFFLINE=1` в `config.py`, `resolve_embedding_model()` никогда не возвращает HF repo-id). Смена модели ⇒ пересборка индекса.
- Векторное хранилище — Chroma (`chroma_db/`); переиндексация без удаления папки (под FileLock). Авто-reindex по вотчеру `data/docs/`.
- Загрузчики: PDF (pypdf, постранично → metadata.page), DOCX (python-docx), TXT (авто-кодировка). Добавление документа → полная переиндексация.
- Для генерации обязателен ключ GigaChat; retrieval работает и без него, но ответа не будет.

## 5. Ссылки на Уровень 2
- Архитектура и поток данных: docs/ARCHITECTURE.md
- Деплой в Yandex Cloud: docs/DEPLOY.md
- API и интеграция GigaChat: docs/API.md

---
Если структура проекта изменилась в ходе сессии — обнови карту и docs/.
