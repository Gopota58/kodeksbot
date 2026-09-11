# ⚖️ КодексБот — юридический RAG-ассистент на GigaChat

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/LLM-GigaChat%20%7C%20Сбер-green)](https://developers.sber.ru/gigachat)
[![Vector Store](https://img.shields.io/badge/Vector%20Store-Chroma-ff6f61)](https://www.trychroma.com/)
[![Retrieval](https://img.shields.io/badge/Retrieval-Hybrid%20%2B%20RRF-9cf)](docs/ARCHITECTURE.md)
[![Reranker](https://img.shields.io/badge/Reranker-MiniLM%20%2F%20jina-ff69b4)](docs/ARCHITECTURE.md)
[![Deploy](https://img.shields.io/badge/Deploy-Yandex%20Cloud-orange)](docs/DEPLOY.md)
[![RAG](https://img.shields.io/badge/RAG-FastAPI%20%2B%20LangChain-005571?logo=fastapi&logoColor=white)](app.py)
[![CI](https://github.com/Gopota58/kodeksbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Gopota58/kodeksbot/actions/workflows/ci.yml)

**КодексБот** отвечает на вопросы по российскому законодательству строго на основе
загруженного корпуса (26 кодексов РФ) и **всегда цитирует источник** — кодекс, статью и
фрагмент. Поиск работает на гибридном ретривере (векторный семантический поиск + BM25),
а генерация — на российской языковой модели **GigaChat** (Сбер). Проект задеплоен в
**Yandex Cloud** и работает целиком в российском контуре.

> Под капотом переиспользуется проверенное RAG-ядро (форк проекта «Котобаза»):
> offline-first, model-agnostic, с гибридным ретривером и оценкой качества.

---

## ✨ Возможности

- 🔍 **Гибридный поиск** — векторный cosine (Chroma) + BM25/TF-IDF, слияние через
  Reciprocal Rank Fusion (RRF). Точнее чистого cosine на русском языке.
- 🏆 **Reranker поверх гибридной выдачи** — подключаемая модель переранжирования
  (`rerank_model`) переупорядочивает кандидатов и поднимает релевантный фрагмент в топ,
  повышая точность цитирования. По умолчанию — лёгкий и быстрый `all-MiniLM-L6-v2`
  (bi-encoder, CPU-friendly); опционально — мультиязычный cross-encoder
  `jina-reranker-v2-base-multilingual` для максимального качества на GPU.
- 📚 **Обязательное цитирование** — под каждым ответом блок «Источники» с карточками
  (кодекс / страница / фрагмент + кнопка «Копировать фрагмент»).
- 🧠 **Локальные русскоязычные эмбеддинги** — `Giga-Embeddings-instruct-480M` (Сбер,
  офлайн, без обращения к сети).
- 🤖 **Генерация на GigaChat** (Сбер) через `langchain-gigachat` — SDK сам делает OAuth,
  обновляет токен и ведёт retry/SSL.
- 📄 **Загрузка своих документов** — PDF / DOCX / TXT через drag-&-drop прямо в веб-интерфейсе.
- 🌗 **Тёмный веб-интерфейс** в «юридическом» стиле (золото/бордовый), адаптивный,
  с историей диалога в `localStorage` (XSS-safe рендер).
- 💬 **Telegram-бот** и десктоп-клиент — опционально (включены в репозиторий).
- ☁️ **Деплой в Yandex Cloud** — Docker Compose на Compute VM, CPU-only.

---

## 🖥 Демо

Веб-интерфейс (тёмная «юридическая» тема, обязательное цитирование источников):

![Веб-интерфейс КодексБот](docs/assets/ui.png)

> Живой демо-сервер: http://89.169.185.183:8000/

---

## 🏗 Архитектура

```mermaid
flowchart LR
    U[Пользователь<br/>веб-UI / Telegram] -->|POST /ask| API[FastAPI<br/>app.py]
    API --> R[RAGEngine<br/>rag/engine.py]
    R --> Q[Query Expansion<br/>юр-синонимы]
    Q --> HY[Гибридный ретривер]
    HY --> V[(Chroma<br/>вектор cosine)]
    HY --> K[BM25 / TF-IDF]
    V --> RRF[RRF-слияние<br/>+ порог 1.35]
    K --> RRF
    RRF --> RR[Reranker<br/>переранжирование кандидатов]
    RR --> LLM[GigaChat-2 · Сбер<br/>langchain-gigachat]
    LLM -->|ответ + цитаты| API
    API -->|источники| U

    subgraph Индексация
        D[data/docs/*.pdf,docx,txt] --> ING[ingest.py]
        ING --> EMB[Giga-Embeddings-480M<br/>строго офлайн]
        EMB --> V
    end
```

Подробнее — в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 🚀 Быстрый старт

### Локально

```bash
# 1. Зависимости
pip install -r requirements.txt

# 2. Конфиг (секреты — только в .env, он в .gitignore)
cp .env.example .env
#   открой .env и впиши LLM_API_KEY (Authorization key из developers.sber.ru)

# 3. Построить индекс из data/docs/
python ingest.py

# 4. Запустить сервер
uvicorn app:app --port 8000
#    открыть http://localhost:8000
```

Проверка чата:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 88888888" \
  -d "{\"question\":\"Какая ответственность за задержку зарплаты?\"}"
```

### Docker

```bash
cp .env.example .env          # заполнить LLM_API_KEY
docker compose up --build
```

> В Docker-контейнере эмбеддинги считаются на CPU (`EMBED_DEVICE=cpu`), torch
> собирается в CPU-варианте, чтобы влезть в малый объём памяти/диска.

---

## ☁️ Деплой в Yandex Cloud (Compute VM)

Полное описание — в [docs/DEPLOY.md](docs/DEPLOY.md). Кратко:

1. Compute VM **Ubuntu 24.04**, 2 vCPU / 4 ГБ / 30 ГБ, публичный IP.
2. Установить **Docker + Docker Compose**.
3. Перенести проект (готовые `models/` ~1.3 ГБ + `chroma_db/` + `data/docs/`),
   выключить вотчер реиндексации.
4. `Dockerfile` собирается с `python:3.12-slim` (numpy 2.5.2 требует ≥3.12),
   `torch==2.14.0+cpu` + индекс PyTorch CPU, `requirements.txt` — строго в UTF-8.
5. Переменные: `EMBED_DEVICE=cpu`, `GIGACHAT_CA_BUNDLE_FILE=/app/certs/Russian_Trusted_Root_CA.pem`.
6. `docker compose up -d`; открыть TCP 8000 в группе безопасности.

Проверено end-to-end: `/health` → `{"status":"ok"}`, `/ask` отвечает со ссылками на статьи
ТК РФ / ГК РФ.

---

## ⚙️ Конфигурация (`.env`)

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `API_KEY` | доступ к `/ask` и админ-эндпоинтам | `88888888` (смените в проде!) |
| `LLM_PROVIDER` | провайдер LLM | `gigachat` |
| `LLM_API_KEY` | Authorization key из консоли GigaChat | — |
| `LLM_MODEL` | id модели (см. `GET /v1/models`) | `GigaChat-2` |
| `GIGACHAT_BASE_URL` | эндпоинт GigaChat | `https://api.giga.chat/v1` |
| `GIGACHAT_VERIFY_SSL_CERTS` | проверка SSL | `true` |
| `GIGACHAT_CA_BUNDLE_FILE` | российский root-CA (нужен на Windows/в облаке) | — |
| `EMBED_PROVIDER` | провайдер эмбеддингов | `local` |
| `EMBEDDING_MODEL_ID` / `model_dir` | локальная модель эмбеддингов | `Giga-Embeddings-instruct-480M-0826` |
| `EMBED_DEVICE` | устройство эмбеддингов | `cuda` (локально) / `cpu` (облако) |
| `RERANK_MODEL` | путь к модели-рерanker (SentenceTransformer bi-encoder, напр. `all-MiniLM-L6-v2`, или cross-encoder jina); пусто — без переранжирования | — |
| `ALLOWED_ORIGINS` | CORS (через запятую, `*` — все) | `*` |

Полный список эндпоинтов и нюансы интеграции GigaChat — в [docs/API.md](docs/API.md).

---

## 📂 Структура проекта

```
kodeksbot/
├── app.py                 # FastAPI: /ask, /ingest, /upload, /documents, /health
├── config.py              # настройки из .env (pydantic-settings)
├── ingest.py              # CLI: построение индекса Chroma из data/docs/
├── rag/engine.py          # ядро RAG: эмбеддинги, Chroma, гибридный ретривер, GigaChat
├── static/index.html      # тёмный веб-чат (vanilla JS/CSS, без сборщиков)
├── bot.py / desktop_client.py  # Telegram-бот / десктоп (опц.)
├── data/docs/             # корпус: 26 кодексов РФ (PDF/DOCX)
├── certs/                 # Russian Trusted Root CA (для SSL GigaChat)
├── docs/                  # документация (Уровень 2)
├── tests/                 # pytest-набор
├── evaluation/            # golden-датасет и метрики качества
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml
```

---

## 🧪 Оценка качества

Каталог `evaluation/` содержит golden-датасет (`golden_dataset.json`), счёт метрик
(`metrics.py`) и сценарий прогона (`run_eval.py`). CI прогоняет `ruff` + `pytest` на
каждый push/PR (`ubuntu-latest`, Python 3.12).

---

## 🗺 Дорожная карта

- [x] Гибридный ретривер (вектор + BM25/RRF), порог иррелевантности 1.35.
- [x] Локальные русскоязычные эмбеддинги (Giga-Embeddings-480M, офлайн).
- [x] Генерация на GigaChat + обязательное цитирование источников.
- [x] Деплой в Yandex Cloud (Docker Compose, CPU).
- [x] Reranker поверх гибридной выдачи (по умолчанию `all-MiniLM-L6-v2`, опц. cross-encoder jina).
- [ ] Смена эмбеддинга на `ai-forever/sbert_large_nlu_ru` (выше качество, тяжелее).
- [ ] Расширение корпуса и мультиарендность.

---

## 🔒 Безопасность

- Реальный `.env` **не коммитится** (в `.gitignore`); секреты — только в переменных
  окружения или Yandex Lockbox, никогда в образе.
- `API_KEY` по умолчанию `88888888` — **обязательно смените** на продакшн-сервере.
- GigaChat-ключ используется только для получения OAuth-токена SDK; сырой ключ никуда
  не уходит как Bearer.

---

## 📄 Лицензия

[MIT](LICENSE) © 2026 Gopota58.

---

## 🔗 Ссылки

- [Архитектура](docs/ARCHITECTURE.md)
- [Деплой в Yandex Cloud](docs/DEPLOY.md)
- [API и интеграция GigaChat](docs/API.md)
- [AGENTS.md](AGENTS.md) — карта проекта для агентов
