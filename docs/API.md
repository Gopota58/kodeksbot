# API и интеграция GigaChat

## Переменные окружения (.env)
```
API_KEY=88888888                         # смените в проде!
LLM_PROVIDER=gigachat
LLM_API_KEY=<Authorization key из developers.sber.ru>
LLM_MODEL=GigaChat-2
GIGACHAT_BASE_URL=https://api.giga.chat/v1
GIGACHAT_VERIFY_SSL_CERTS=true
# GIGACHAT_CA_BUNDLE_FILE=C:\...\russian_trusted_root_ca.crt  (нужен на Windows/в облаке)
EMBED_PROVIDER=local
EMBEDDING_MODEL_ID=Giga-Embeddings-instruct-480M-0826
EMBED_DEVICE=cpu                          # cuda локально / cpu в облаке
ALLOWED_ORIGINS=*
```
Шаблон — `.env.example`. Реальный `.env` не коммитится.

## Эндпоинты (FastAPI, `app.py`)
- `GET /` → отдаёт `static/index.html` (веб-чат).
- `GET /health` → `{"status":"ok"}`.
- `POST /ask` (JSON `{"question": str}`, заголовок `X-API-Key`) →
  `{"answer": str, "sources": [{"source": str, "page": int|null, "snippet": str}]}`.
- `POST /ingest` → принудительная переиндексация `data/docs/`.
- `POST /upload` (multipart, `.pdf`/`.docx`/`.txt`) → добавляет файл + переиндексация.
- `GET /documents` → список файлов корпуса.
- `DELETE /documents/{filename}` → удаление + переиндексация.

## GigaChat (`rag/engine.py` → `build_llm`)
- `from langchain_gigachat import GigaChat`; SDK сам меняет `LLM_API_KEY` на `access_token`
  (OAuth) и обновляет его.
- Модель обязательна (`LLM_MODEL`); актуальные id — `GET https://api.giga.chat/v1/models`
  (GigaChat-2 — бесплатный Lite-тариф, GigaChat-Pro / GigaChat-Max — платные).
- На Windows нужен CA-бандл `GIGACHAT_CA_BUNDLE_FILE` (российский root-CA с gosuslugi.ru/crt)
  либо `GIGACHAT_VERIFY_SSL_CERTS=false` для dev.
- Не используйте reasoning-модели без отключения thinking — вернут пустой ответ.

## Эмбеддинги (`rag/engine.py` → `build_embeddings`)
- `local` (по умолчанию): локальная русскоязычная `Giga-Embeddings-instruct-480M-0826` (Сбер,
  Qwen3Bidirectional, 1024-dim, bf16, SentenceTransformer с `trust_remote_code`), загружается
  **строго офлайн** (`HF_HUB_OFFLINE=1`). Instruct-префикс `query`/`document`.
  Для максимального качества возможна `ai-forever/sbert_large_nlu_ru` (тяжелее, качать вручную).
- `api`: OpenAI-совместимый endpoint (напр. nomic-embed-text в LM Studio).
- Смена модели ⇒ пересборка индекса (`python ingest.py`).
