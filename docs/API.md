# API и интеграция GigaChat

## Переменные окружения (.env)
```
API_KEY=kb_pub_...                        # публичный: только /ask (то же значение в static/index.html)
ADMIN_API_KEY=<случайная строка>          # админ: /upload, /ingest, /documents, DELETE
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=30                  # /ask на один IP
RATE_LIMIT_ADMIN_PER_MINUTE=10            # админ-эндпоинты на один IP
LLM_PROVIDER=gigachat
LLM_API_KEY=<Authorization key из developers.sber.ru>
LLM_MODEL=GigaChat-2
GIGACHAT_BASE_URL=https://api.giga.chat/v1
GIGACHAT_VERIFY_SSL_CERTS=true
# GIGACHAT_CA_BUNDLE_FILE=C:\...\russian_trusted_root_ca.crt  (нужен на Windows/в облаке)
EMBED_PROVIDER=local
EMBEDDING_MODEL_ID=Giga-Embeddings-instruct-480M-0826
EMBED_DEVICE=cpu                          # cuda локально / cpu в облаке
ENABLE_HYDE=false                         # true: +1 вызов LLM на запрос, лучше семантика
RERANK_MODEL=models/all-MiniLM-L6-v2      # пусто — reranker выключен
ALLOWED_ORIGINS=*
```
Шаблон — `.env.example`. Реальный `.env` не коммитится.

## Эндпоинты (FastAPI, `app.py`)

| Метод и путь | Ключ | Назначение |
|---|---|---|
| `GET /` | — | редирект на `static/index.html` (веб-чат) |
| `GET /health` | — | `{"status":"ok"}` |
| `GET /favicon.ico` | — | редирект на `/static/favicon.svg` |
| `POST /ask` | `API_KEY` (публичный) | JSON `{"question": str}` → `{"answer": str, "sources": [{"source", "page", "snippet"}]}` |
| `POST /ingest` | `ADMIN_API_KEY` | принудительная переиндексация `data/docs/` |
| `POST /upload` | `ADMIN_API_KEY` | multipart `.pdf`/`.docx`/`.txt` → файл + переиндексация |
| `GET /documents` | `ADMIN_API_KEY` | список файлов корпуса |
| `DELETE /documents/{filename}` | `ADMIN_API_KEY` | удаление + переиндексация |

### Аутентификация
Ключ передаётся в заголовке `X-API-Key`. Ключи **разделены по зонам ответственности**:

- `API_KEY` — публичный, открывает только `/ask`. Он намеренно попадает в открытый
  `static/index.html`, поэтому секретом не является: его видит любой посетитель демо.
- `ADMIN_API_KEY` — всё, что меняет состояние (`/upload`, `/ingest`, `/documents`, `DELETE`).
  В статику не попадает; в веб-UI вводится вручную кнопкой 🔑 и хранится в `localStorage`.
  Если не задан — используется `API_KEY` (обратная совместимость, локальная разработка).

Неверный или отсутствующий ключ → `401`.

### Rate limiting
Скользящее окно 60 секунд на пару (IP, группа эндпоинтов). `/ask` — 30 запросов/мин,
админ-эндпоинты — 10/мин. Превышение → `429 Too Many Requests` с заголовком
`Retry-After` (секунды) и телом `{"detail": "..."}`.

CORS-preflight (`OPTIONS`) в лимит **не** засчитывается — он не доходит до LLM, а браузер
шлёт его перед каждым кросс-доменным POST.

Ограничение считается **до** проверки ключа, поэтому защищает и от перебора.
Состояние — в памяти процесса: для одной реплики достаточно, при горизонтальном
масштабировании нужен общий стор (Redis).

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
