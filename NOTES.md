## 2026-09-08 — RAG «КодексБот»: индексация 26 кодексов, локальные эмбеддинги, GigaChat, чиним гибридный ретривер

### Сделано (Done)
- Загружена локальная модель `cointegrated/rubert-tiny2` в `E:\Project\models\rubert-tiny2`
  (веса model.safetensors + pytorch_model.bin + адаптер; грузится СТРОГО офлайн через
  `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`, прописанные в `config.py`).
- `config.py`: `resolve_embedding_model()` всегда отдаёт локальный путь и падает, если модели нет
  (никогда не уходит в сеть/скачивание).
- Корпус 26 кодексов РФ проиндексирован: `python ingest.py` → 6493 страницы → **41110 чанков**
  → Chroma `kodeksbot` (~400 МБ в `chroma_db/`).
- `.env` создан (gitignored): GigaChat-провайдер, ключ (base64 client_id:client_secret),
  `GIGACHAT_CA_BUNDLE_FILE` = `E:\Project\certs\Russian_Trusted_Root_CA.pem`.
  GigaChat дымовой тест прошёл (вернул ответ + token_usage).
- **Исправлен гибридный ретривер** (`rag/engine.py`):
  1. `_build_keyword_index` читал всю коллекцию одним `coll.get()` → SQLite «too many SQL variables»
     на 41k чанках → BM25-индекс не строился (vectorizer=None) → только векторный поиск.
     Исправлено пагинацией `limit=5000, offset=`.
  2. `_keyword_search` теперь отдаёт Document с метаданными (source/page).
- Итог: гибридный поиск (вектор cosine + TF-IDF/BM25 → RRF) работает; для «неустойка по ГК РФ»
  возвращает ГК РФ ч.1 ст. 330-395, Жилищный кодекс и т.д.

### Важное уточнение по «багу» (НЕ баг сервера!)
- При тестах через PowerShell `Invoke-RestMethod` `/ask` отдавал Конституцию. Корень — **кодировка
  тела запроса**: PowerShell отправляет кириллицу не в UTF-8, поэтому запрос доходит как mojibake,
  TF-IDF не находит «неустойка» → `KW=[]` → только векторный поиск → Конституция.
- Чистый Python-клиент (`requests`, строгий UTF-8) возвращает ГК корректно. Сервер и пайплайн верны.
- Вывод в консоли PowerShell везде mojibake (UTF-8 как latin-1) — это ТОЛЬКО отображение, данные корректны.

### Следующие шаги (Next Steps)
1. Красивый веб-интерфейс (обсуждалось отдельно) — `static/` пока минимальный.
2. Поднять качество ранжирования: дообучить/сменить эмбеддинг на `ai-forever/sbert_large_nlu_ru`
   (точнее rubert-tiny2) — потребует пересборки индекса.
3. При необходимости добавить reranker (cross-encoder) поверх гибридной выдачи.
4. (Опц.) Вынести корпус `data/docs/` в git-lfs или отдельное хранилище — он не в git.
5. Ротировать секреты, которые попали в чат (GitHub/HF/GigaChat-ключи) — они упоминались в переписке.

### Грабли и находки (Gotchas)
- HF LFS CDN (`cdn-lfs.huggingface.co`, `us.aws.cdn.hf.co`) заблокирован в этой сети — модель
  качалась на другой машине вручную. Офлайн-режим обязателен.
- GigaChat на Windows требует российский root-CA (`gosuslugi.ru/crt` → PEM), иначе SSL-ошибка.
- `chroma_db` + 41k чанков: прямой `coll.get()` без пагинации роняет SQLite лимитом переменных.
- **Кириллические запросы в HTTP нужно слать строго в UTF-8**; иначе keyword-поиск молча пустеет
  и система деградирует до векторного (семантически «ближайшего», т.е. Конституции).
- Запуск только через venv `C:\rag_project\venv\Scripts\python.exe` из `E:\Project`.
  Команда: `python -m uvicorn app:app --port 8000` (watcher можно отключить `RAG_DISABLE_WATCHER=1`).
- Всегда отвечать пользователю по-русски; без явной команды не менять код/конфиги/git.
