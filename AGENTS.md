# AGENTS.md — КодексБот (юридический RAG-ассистент)

## 1. Суть проекта
RAG-ассистент для юридических документов: отвечает на вопросы по пользовательским
файлам (PDF/DOCX/TXT) с обязательным цитированием источника. Стек: Python +
FastAPI + LangChain + Chroma (векторная БД) + GigaChat (Сбер, через langchain-gigachat)
+ локальные эмбеддинги sentence-transformers для генерации. Статус: каркас собран на базе фреймворка «Котобаза» (FastAPI+LangChain+Chroma); корпус из 26 кодексов РФ загружен в data/docs/.

## 2. Карта папок
- `rag/engine.py` — ядро RAG: эмбеддинги, Chroma, гибридный ретривер (вектор+BM25→RRF), **reranker поверх выдачи** (MiniLM по умолч., опц. jina cross-encoder), GigaChat, reindex, watcher
- `app.py` — FastAPI: `/ask` (+источники), `/ingest`, `/upload`, `/documents`, `/health`, CORS, статикa
- `config.py` — настройки из `.env` (pydantic-settings): LLM / эмбеддинги / Chroma
- `ingest.py` — CLI: `python ingest.py` (строит индекс Chroma из data/docs/)
- `static/index.html` — веб-чат (один файл, vanilla JS/CSS, без сборщиков): тёмный «юридический»
  дизайн, обязательное цитирование источников (карточки кодекс/стр./фрагмент), drag&drop загрузка
  (`/upload`), список документов (`/documents`), переиндексация (`/ingest`), статус сервера (`/health`).
  XSS-safe рендер (`textContent`/`esc()`), история в `localStorage`.
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
- Эмбеддинги — локальная **Giga-Embeddings-instruct-480M-0826** (Сбер, 1024-dim, bf16,
  SentenceTransformer-формат с `trust_remote_code=True` → `modeling_gigarembed.py`). Лежит в
  `models/Giga-Embeddings-instruct-480M-0826`, грузится **строго локально без сети**
  (`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` в `config.py`). Instruct-промпты: query-префикс
  `Instruct: Given a query, retrieve relevant passages\nQuery: ` (документы без префикса).
  Смена модели ⇒ пересборка индекса.
- Эмбеддинги считаются на **GPU локально** (`embed_device` берётся из `EMBED_DEVICE`, по умолчанию
  `"cuda"`; в Docker-деплое `EMBED_DEVICE=cpu`). Для reindex на GPU нужна CUDA-сборка torch
  (`torch==2.11.0+cu128`). VRAM ~3.5 ГБ под Giga-480M.
- **Деплой в Yandex Cloud (Docker, CPU):** `Dockerfile` собирается с `python:3.12-slim` (не 3.11 —
  замороженный `requirements.txt` под 3.12, `numpy==2.5.2` требует ≥3.12). `requirements.txt` должен быть
  **строго в UTF-8** (кириллические комментарии в CP1251 роняют `pip install`). torch запинен как
  `torch==2.14.0+cpu` + `--extra-index-url https://download.pytorch.org/whl/cpu` (иначе тянется CUDA
  ~3 ГБ, что убивает 4 ГБ RAM/30 ГБ диска ВМ). GigaChat из ВМ работает (RF-CA нативный, либо
  `GIGACHAT_CA_BUNDLE_FILE=/app/certs/Russian_Trusted_Root_CA.pem`).
- Порог иррелевантности `_MAX_IRRELEVANT_DISTANCE = 1.35` в `rag/engine.py` (косинусная дистанция
  Chroma: выше — не релевантно; откалибровано под Giga: REL≈0.88..1.33, NOISE≈1.37..1.65).
  Старый порог 0.60 (для rubert-tiny2) не годится — отсекал все релевантные запросы.
- Векторное хранилище — Chroma (`chroma_db/`); переиндексация без удаления папки (под FileLock). Авто-reindex по вотчеру `data/docs/`.
- Загрузчики: PDF (pypdf, постранично → metadata.page), DOCX (python-docx), TXT (авто-кодировка). Добавление документа → полная переиндексация.
- Для генерации обязателен ключ GigaChat; retrieval работает и без него, но ответа не будет.

## 5. Ссылки на Уровень 2
- Архитектура и поток данных: docs/ARCHITECTURE.md
- Деплой в Yandex Cloud: docs/DEPLOY.md
- API и интеграция GigaChat: docs/API.md

---
Если структура проекта изменилась в ходе сессии — обнови карту и docs/.
