# Архитектура КодексБот

## Обзор
Переиспользуемый RAG-движок (форк проекта «Котобаза»): **ingest → index → retrieve → generate**.
Генерация — GigaChat (Сбер) через `langchain-gigachat`; эмбеддинги — локальная
русскоязычная модель `cointegrated/rubert-tiny2` (тянется из HF при 1-м запуске);
векторное хранилище — Chroma (persistent).

## Поток данных

### Индексация (offline / при старте / по вотчеру)
```
data/docs/*.pdf|docx|txt
  → rag/engine.py: load_all_documents()
      - PDF:   pypdf (постранично, metadata.page)
      - DOCX:  python-docx
      - TXT:   авто-кодировка (chardet)
  → RecursiveCharacterTextSplitter (chunk≈1000, overlap≈150)
  → HuggingFaceEmbeddings (rubert-tiny2) → векторы
  → Chroma.from_documents() → chroma_db/ (collection: kodeksbot)
```

### Запрос (online)
```
Пользователь (static/index.html) → POST /ask {question}
  → rag/engine.py: ask_with_sources()
      - _expand_query()        (синонимы юр-терминов для лучшего retrieval)
      - _retrieve_hybrid():    векторный cosine + BM25/TF-IDF → RRF (top-k=8)
      - build_llm() → GigaChat (OAuth-токен через SDK)
      - промпт с обязательным цитированием источника
  → {answer, sources:[{source, page, snippet}]}
```

## Модули
- `config.py` — настройки из `.env` (pydantic-settings).
- `rag/engine.py` — `RAGEngine`: эмбеддинги, Chroma, гибридный ретривер, LLM, reindex, watcher.
- `app.py` — FastAPI: `/ask`, `/ingest`, `/upload`, `/documents`, `/health`.
- `ingest.py` — CLI: `python ingest.py`.
- `static/` — веб-чат (vanilla JS); красивый UI обсуждается отдельно.

## Решения по моделям
- **Эмбеддинги:** локальная русскоязычная `cointegrated/rubert-tiny2` (быстрая, ~90 МБ, тянется из HF при 1-м запуске).
  Для максимального качества — `ai-forever/sbert_large_nlu_ru` (~1.2 ГБ). `all-MiniLM-L6-v2` (из запасов фреймворка)
  слабее для русского и не используется по умолчанию.
- **LLM:** GigaChat (`LLM_PROVIDER=gigachat`, `LLM_MODEL` — актуальный id из `GET /v1/models`).
- **Ретривер:** гибридный вектор + BM25 с RRF — точнее чистого cosine на русском.

## Состояние
- [x] корпус документов загружен: 26 кодексов РФ в `data/docs/`
- [x] каркас собран на базе фреймворка «Котобаза» (Chroma + GigaChat)
- [ ] требуется `.env` с GigaChat-ключом для запуска генерации
- [ ] веб-UI будет переработан (обсуждается красивый интерфейс)
- [ ] деплой в Yandex Cloud не реализован
