# Деплой КодексБот в Yandex Cloud

Проект задеплоен на **Compute VM** (Docker Compose). Это проще, чем Serverless Containers,
когда нужен постоянный векторный индекс Chroma и локальные эмбеддинги: диск — постоянный,
индекс пересобирается один раз и лежит на SSD.

> Альтернатива — Serverless Containers + API Gateway — описана в конце (с нюансами про
> эфемерную ФС и эмбеддинги).

## Вариант А: Compute VM + Docker Compose (используется)

### 1. Создать ВМ
- Образ: **Ubuntu 24.04 LTS**.
- Конфигурация: **2 vCPU / 4 ГБ RAM / 30 ГБ SSD**, публичный IP.
- Открыть в группе безопасности TCP **8000** (источник — свой IP или `0.0.0.0/0`).

### 2. Установить Docker + Compose
```bash
sudo apt update && sudo apt install -y docker.io
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo systemctl enable --now docker
```

### 3. Перенести проект на ВМ
Переносим **готовые** тяжёлые артефакты (не качаем заново):
- `models/Giga-Embeddings-instruct-480M-0826/` (~1.3 ГБ)
- `chroma_db/` (~603 МБ, уже построенный индекс)
- `data/docs/` (~136 МБ, корпус кодексов)
- `certs/Russian_Trusted_Root_CA.pem` (для SSL GigaChat)

Вотчер авто-реиндексации на сервере **выключен** (`RAG_DISABLE_WATCHER=1`).

### 4. Собрать и запустить контейнер
Ключевые параметры образа (см. `Dockerfile` / `docker-compose.yml`):
- База `python:3.12-slim` (numpy 2.5.2 из `requirements.txt` требует ≥3.12; на 3.11 билд падает).
- `requirements.txt` — **строго в UTF-8** (кириллические комментарии в CP1251 роняют `pip install`).
- `torch==2.14.0+cpu` + `--extra-index-url https://download.pytorch.org/whl/cpu`
  (незапиненный torch в Linux тянет CUDA ~3 ГБ → OOM/диск).
- Переменные окружения контейнера:
  - `EMBED_DEVICE=cpu`
  - `GIGACHAT_CA_BUNDLE_FILE=/app/certs/Russian_Trusted_Root_CA.pem`
  - `LLM_PROVIDER=gigachat`, `LLM_API_KEY`, `LLM_MODEL` (например `GigaChat-2`)
  - `API_KEY` (сменить с дефолтного `88888888`!)

```bash
docker compose up -d --build
```

### 5. Проверить
```bash
curl http://localhost:8000/health      # {"status":"ok"}
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" -d '{"question":"Зарплата при увольнении"}'
```
End-to-end проверено: «зарплата при увольнении» → ст. 140 ТК РФ; «долг по кредиту» →
ст. 812/813 ГК РФ (8 источников). Веб-UI отдаётся с диска (`static/index.html`).

> Опционально: `sudo systemctl enable docker` — автостарт контейнера после перезагрузки ВМ.

## Вариант Б: Serverless Containers + API Gateway (альтернатива)
Платим за вызовы, нет постоянно работающей VM.
1. `docker build -t cr.yandex/<registry-id>/kodeksbot:latest .`
2. `docker push cr.yandex/<registry-id>/kodeksbot:latest`
3. `yc serverless container create --name kodeksbot --image cr.yandex/<registry-id>/kodeksbot:latest`
4. API Gateway, проксирующий на container URL.
5. Переменные окружения контейнера: `LLM_PROVIDER=gigachat`, `LLM_API_KEY`, `LLM_MODEL`,
   `GIGACHAT_CA_BUNDLE_FILE`, `EMBED_PROVIDER=local`, `EMBEDDING_MODEL_ID`, `API_KEY`.
   Секреты — через **Yandex Lockbox**, не в образе.

### Важно про Chroma и эмбеддинги в serverless
- Индекс Chroma — файл (`chroma_db/`). В serverless ФС эфемерна: индекс надо либо собирать
  при старте контейнера из `data/docs/` (смонтированный том), либо перейти на управляемую БД
  (Yandex Managed PostgreSQL + pgvector).
- Локальные эмбеддинги тянут модель (~1.3 ГБ) при первом старте — учитывайте лимиты
  памяти/времени контейнера.

## Локальный запуск (без Docker)
```bash
pip install -r requirements.txt
cp .env.example .env   # заполнить GigaChat-ключ
python ingest.py
uvicorn app:app --port 8000
```
