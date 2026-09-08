# Архитектура КодексБот

## Обзор
Переиспользуемое RAG-ядро (форк проекта «Котобаза»): **ingest → index → retrieve → generate**.
Генерация — **GigaChat** (Сбер) через `langchain-gigachat` (OAuth-токен, SSL, retry);
эмбеддинги — **локальная русскоязычная модель `Giga-Embeddings-instruct-480M-0826`**
(Сбер, Qwen3Bidirectional, 1024-dim, bf16, SentenceTransformer с `trust_remote_code`),
загружается **строго офлайн** (`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` в `config.py`).
Векторное хранилище — **Chroma** (persistent).

## Поток данных

### Индексация (offline / при старте / по вотчеру)
```
data/docs/*.pdf|docx|txt
  → rag/engine.py
      - PDF:   pypdf / fitz (PyMuPDF, фолбэк), постранично (metadata.page)
      - DOCX:  python-docx
      - TXT:   авто-кодировка (chardet)
  → RecursiveCharacterTextSplitter (chunk≈400, overlap≈60)
  → InstructGigaEmbeddings (Giga-Embeddings-480M, offline)
       * embed_documents → prompt_name="document"
       * embed_query     → prompt_name="query" (instruct-префикс только у query)
  → Chroma.from_documents() → chroma_db/ (collection: kodeksbot, ~40k+ чанков)
```

### Запрос (online)
```
Пользователь (static/index.html) → POST /ask {question}
  → rag/engine.py
      - _expand_query()        (юр-синонимы: зарплата↔заработная плата и т.п.)
      - _retrieve_hybrid():    векторный cosine (Chroma) + BM25/TF-IDF → RRF (top-k=8)
      - фильтр иррелевантности (_MAX_IRRELEVANT_DISTANCE = 1.35, шкала Chroma)
      - build_llm() → GigaChat (OAuth-токен через SDK)
      - промпт с обязательным цитированием источника
  → {answer, sources:[{source, page, snippet}]}
```

## Модули
- `config.py` — настройки из `.env` (pydantic-settings): LLM / эмбеддинги / Chroma / GigaChat.
- `rag/engine.py` — `RAGEngine`: эмбеддинги, Chroma, гибридный ретривер, LLM, reindex, watcher.
- `app.py` — FastAPI: `/ask`, `/ingest`, `/upload`, `/documents`, `/health`, CORS, статика.
- `ingest.py` — CLI: `python ingest.py`.
- `static/index.html` — веб-чат (один файл, vanilla JS/CSS): тёмный «юридический» дизайн,
  обязательное цитирование источников (карточки кодекс/стр./фрагмент + «Копировать»),
  drag&drop загрузка, список документов, переиндексация, статус сервера, история в `localStorage`, XSS-safe.
- `bot.py` / `desktop_client.py` — Telegram-бот / десктоп (опц., из фреймворка).

## Решения по моделям
- **Эмбеддинги:** локальная `Giga-Embeddings-instruct-480M-0826` (Сбер, instruct, 1024-dim, офлайн).
  Замена модели ⇒ пересборка индекса (`python ingest.py`). Для максимального качества
  возможна `ai-forever/sbert_large_nlu_ru` (~1.2 ГБ, надо качать вручную — HF LFS CDN заблокирован).
- **LLM:** GigaChat (`LLM_PROVIDER=gigachat`, `LLM_MODEL` — актуальный id из `GET /v1/models`,
  по умолчанию `GigaChat-2`). Сырой `ChatOpenAI` не годится — GigaChat требует OAuth, который
  делает SDK (`langchain-gigachat`).
- **Ретривер:** гибридный вектор + BM25 с RRF — точнее чистого cosine на русском.
  Порог иррелевантности **1.35** откалиброван под Giga (REL≈0.88..1.33, NOISE≈1.37..1.65).

## Состояние
- [x] корпус: 26 кодексов РФ в `data/docs/` (PDF/DOCX)
- [x] локальные русскоязычные эмбеддинги (Giga-Embeddings-480M, офлайн)
- [x] гибридный ретривер (вектор + BM25/RRF) + query expansion
- [x] генерация на GigaChat + обязательное цитирование
- [x] тёмный веб-UI с цитированием источников
- [x] деплой в Yandex Cloud (Compute VM, Docker Compose, CPU) — см. [DEPLOY.md](DEPLOY.md)
- [ ] reranker (cross-encoder) поверх гибридной выдачи
