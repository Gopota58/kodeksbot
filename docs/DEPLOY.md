# Деплой КодексБот в Yandex Cloud

## Вариант: Serverless Containers + API Gateway
Платим за вызовы, нет постоянно работающей VM.

1. Сборка образа: `docker build -t cr.yandex/<registry-id>/кодексбот:latest .`
2. Пуш: `docker push cr.yandex/<registry-id>/кодексбот:latest`
3. `yc serverless container create --name кодексбот --image cr.yandex/<registry-id>/кодексбот:latest`
4. API Gateway, проксирующий на container URL.
5. Переменные окружения контейнера: `LLM_PROVIDER=gigachat`, `LLM_API_KEY`, `LLM_MODEL`,
   `GIGACHAT_CA_BUNDLE_FILE`, `EMBED_PROVIDER=local`, `EMBEDDING_MODEL_ID`, `API_KEY`.
   Секреты — через Yandex Lockbox, не в образе.

## Важно про Chroma и эмбеддинги в serverless
- Индекс Chroma — файл (`chroma_db/`). В serverless ФС эфемерна: индекс надо либо
  собирать при старте контейнера из `data/docs/` (смонтированный том), либо перейти
  на управляемую БД (Yandex Managed PostgreSQL + pgvector).
- Локальные эмбеддинги тянут модель (~90 МБ–1.2 ГБ) при первом старте — учитывайте
  лимиты памяти/времени контейнера.

## Локальный запуск
```
pip install -r requirements.txt
cp .env.example .env   # заполнить GigaChat-ключ
python ingest.py
uvicorn app:app --port 8000
```

## Альтернатива: Compute VM
Маленькая VM + Docker Compose — проще отладка и постоянное хранилище, но постоянная стоимость.
