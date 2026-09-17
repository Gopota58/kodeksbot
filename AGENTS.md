# AGENTS.md — КодексБот (юридический RAG-ассистент)

## 1. Суть проекта
RAG-ассистент для юридических документов: отвечает на вопросы по пользовательским
файлам (PDF/DOCX/TXT) с обязательным цитированием источника. Стек: Python +
FastAPI + LangChain + Chroma (векторная БД) + GigaChat (Сбер, через langchain-gigachat)
+ локальные эмбеддинги sentence-transformers для генерации. Статус: каркас собран на базе фреймворка «Котобаза» (FastAPI+LangChain+Chroma); корпус из 26 кодексов РФ загружен в data/docs/.

## 2. Карта папок
- `rag/engine.py` — ядро RAG: эмбеддинги, Chroma, гибридный ретривер (вектор+BM25→RRF), **reranker поверх выдачи** (MiniLM по умолч., опц. jina cross-encoder), GigaChat, reindex, watcher
- `app.py` — FastAPI: `/ask` (+источники), `/ingest`, `/upload`, `/documents`, `/health`,
  `/favicon.ico`, CORS, статика, **rate limiting** (скользящее окно 60 с на IP) и **два ключа
  доступа**: публичный `API_KEY` (только `/ask`) и `ADMIN_API_KEY` (всё, что меняет состояние)
- `config.py` — настройки из `.env` (pydantic-settings): LLM / эмбеддинги / Chroma /
  ключи доступа / лимиты частоты
- `ingest.py` — CLI: `python ingest.py` (строит индекс Chroma из data/docs/)
- `static/index.html` — веб-чат (один файл, vanilla JS/CSS, без сборщиков): тёмный «юридический»
  дизайн, обязательное цитирование источников (карточки кодекс/стр./фрагмент), drag&drop загрузка
  (`/upload`), список документов (`/documents`), переиндексация (`/ingest`), статус сервера (`/health`),
  кнопка 🔑 для ввода админ-ключа (хранится в `localStorage`, в HTML не попадает).
  XSS-safe рендер (`textContent`/`esc()`), история в `localStorage`.
- `static/favicon.svg` — иконка (весы Фемиды), отдаётся через `/favicon.ico`
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
- Проверка чата: `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" -d "{\"question\":\"...\"}"`
- Проверка админки: `curl -H "X-API-Key: $ADMIN_API_KEY" http://localhost:8000/documents`
- Проверка лимитов: 31 запрос к `/ask` за минуту с одного IP → последний вернёт `429` + `Retry-After`
- Тесты (герметичны, без сети/GPU/корпуса): `pytest -q` — 20 тестов, ~20 с
- Линт: `ruff check .` (обязательный гейт в CI, должен быть чистым)
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
- **Ключи разделены.** `API_KEY` — публичный, открывает только `/ask`, и он же зашит в
  `static/index.html` (значит, не секрет: его видит любой посетитель). `ADMIN_API_KEY` —
  для `/upload`, `/ingest`, `/documents`, `DELETE /documents/{file}`; в статику не попадает.
  Меняя публичный ключ, **обязательно** поправьте константу `API_KEY` в `static/index.html`,
  иначе UI получит 401. Если `ADMIN_API_KEY` пуст — админка открывается публичным ключом
  (только для локальной разработки, на публичном стенде недопустимо).
- **Rate limiting** живёт в памяти процесса (`app.py`, `_rl_hits`) — при нескольких репликах
  нужен общий стор. Считается **до** проверки ключа.
- **ВМ в Yandex Cloud:** swap 2 ГБ в `/etc/fstab` (`vm.swappiness=10`) — страховка от OOM на
  4 ГБ RAM. Диск 30 ГБ забивает Docker build cache (до ~12 ГБ): лечится `docker builder prune -f`
  (только неиспользуемый кэш). Перед пересборкой тегировать рабочий образ как
  `kodeksbot-app:rollback-<дата>` — откат одной командой.
- **Обновление кода на ВМ — ТОЛЬКО через git.** `git fetch origin main && git reset --hard origin/main`
  → `docker compose build` → `docker compose up -d`. Копирование файлов через `scp` (и «сверку по md5»)
  использовать НЕЛЬЗЯ: так копится незаметное расхождение — ВМ уехала на 20+ коммитов назад, при том что
  сервис выглядел рабочим. Перед сбросом: бэкап `.env` (`cp .env .env.bak-$(date +%Y%m%d-%H%M%S)`)
  и тег рабочего образа.
- **`requirements.txt` + `pip install`:** torch запинен как `torch==<ver>+cpu`, а сборки с локальным
  тегом `+cpu` есть **только** на индексе PyTorch. Без `--extra-index-url https://download.pytorch.org/whl/cpu`
  установка падает (на PyPI такого файла нет) — так CI был красным две недели.
- **Тесты обязаны быть герметичными.** `data/docs/` и `chroma_db/` в `.gitignore` — в CI их нет.
  `tests/conftest.py` выставляет env (hash-эмбеддер, временный `CHROMA_DIR`, `DOCS_DIR` → `tests/fixtures/docs/`,
  `RERANK_MODEL=""`, `RAG_DISABLE_WATCHER=1`) **до** импорта `config`. Любой новый тест, которому нужен
  корпус, обязан брать его из фикстур, а не из рабочего `data/docs/`.
- **Reranker:** дефолт — `models/all-MiniLM-L6-v2` (bi-encoder, CPU-friendly). `jina-reranker-v2-base-multilingual`
  точнее, но на CPU неприемлемо медленный — только под GPU. В `docker-compose.yml` переменная
  `RERANK_MODEL` задана явно.
- **Пересборка образа — ~20 мин**, а не секунды: `COPY requirements.txt` входит в кэш-ключ
  `RUN pip install`, и правка даже комментария в манифесте тянет полную переустановку torch.
  `COPY . .` идёт после `pip install`, поэтому правки кода сами по себе дёшевы.
- Первый `/ask` после старта может сбросить соединение (прогрев эмбеддингов + reranker);
  ждать `Application startup complete` (~90 с на ВМ). Сразу после `up -d` контейнер уже `Up`,
  но приложение ещё не готово — опрашивать `/health`, а не верить `docker ps`.
- **Зависшее имя контейнера.** Прерванный `docker compose up -d` (таймаут, Ctrl-C) оставляет в
  демоне запись: следующий `up` падает с `Conflict … name is already in use`, хотя `docker ps -a`
  этого контейнера уже не показывает. `docker rm -f` и `--force-recreate` не помогают —
  лечится `sudo systemctl restart docker` (данные в bind-mounts не страдают).
- **Долгие ssh-команды из Git Bash обрываются по SIGTERM на 120 с.** Сборку и другие долгие
  операции запускать отделённо (`setsid nohup … > /tmp/build.log 2>&1 &`) и забирать вывод позже.
  Ждать окончания сборки по строке `Image … Built` в логе, а НЕ по `pgrep -f buildkit` —
  этот процесс живёт постоянно и совпадение будет всегда.
- Транзиентный сбой сети ВМ: `failed to fetch anonymous token … TLS handshake timeout` к
  `auth.docker.io`. Повторить сборку; рабочий контейнер до успешной проверки новой сборки не удалять.
- `docker exec` без рабочей директории не видит модули проекта — нужен `-w /app`.
  Инлайн-python через ssh ломается на кавычках: класть скрипт heredoc'ом в файл + `docker cp`.

## 5. Ссылки на Уровень 2
- Архитектура и поток данных: docs/ARCHITECTURE.md
- Деплой в Yandex Cloud: docs/DEPLOY.md
- API и интеграция GigaChat: docs/API.md

---
Если структура проекта изменилась в ходе сессии — обнови карту и docs/.
