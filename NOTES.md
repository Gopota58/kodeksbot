## 2026-09-08 (продолжение 3) — Деплой КодексБот в Yandex Cloud (Docker, CPU, бесплатный грант)

### Сделано (Done)
- Полный деплой рабочего стека на ВМ Yandex Cloud (Ubuntu 24.04, 2 vCPU / 4 ГБ / 30 ГБ, публичный IP
  `89.169.185.183`, OS Login `gopota58`). Docker 29.8 + Compose v5.5.1 установлены; проект перенесён по
  SSH (готовые `models/` 1.3 ГБ + `chroma_db/` 603 МБ + `data/docs/` 136 МБ; watcher выключен).
- `Dockerfile`: база `python:3.12-slim` (была 3.11 — падала на `numpy==2.5.2`, требует ≥3.12).
- `requirements.txt`: перекодирован в **UTF-8** (был Windows-1251 → `pip install` падал `UnicodeDecodeError`).
  torch запинен `torch==2.14.0+cpu` + `--extra-index-url https://download.pytorch.org/whl/cpu`
  (иначе тянул CUDA ~3 ГБ → не влезает в 4 ГБ RAM/30 ГБ диска).
- `config.py`: `embed_device` теперь `os.getenv("EMBED_DEVICE","cuda")` (в compose `EMBED_DEVICE=cpu`).
- Контейнер `kodeksbot` **Up**, `/health` → `{"status":"ok"}` (200), веб-UI отдаётся
  (`⚖️ КодексБот — юридический RAG-ассистент`). End-to-end `/ask` на CPU проверен:
  «зарплата при увольнении» → ст. 140 ТК РФ; «долг по кредиту» → ст. 812/813 ГК РФ (8 источников).
- GigaChat из ВМ работает (SSL через `GIGACHAT_CA_BUNDLE_FILE=/app/certs/Russian_Trusted_Root_CA.pem`).

### Грабли
- **Кодировка requirements.txt**: кириллица из `Add-Content -Encoding Default` пишется в CP1251 → pip
  читает как UTF-8 → `UnicodeDecodeError`. Лечится перекодировкой в UTF-8.
- **Python 3.11 vs 3.12**: замороженный `requirements.txt` (numpy 2.5.2 и др.) собран под 3.12 → на
  3.11-slim `pip` не находит версии → билд падает. База образа должна быть `python:3.12-slim`.
- **CUDA-torch на CPU-ВМ**: незапиненный `torch` в Linux тянет CUDA-колёса (~3 ГБ) → OOM/диск. Всегда
  пинить `torch==<ver>+cpu` + PyTorch CPU-индекс для облачного CPU-деплоя.
- **Внешний доступ**: с моего места `http://89.169.185.183:8000` → `HTTP 000`, но с самой ВМ её
  публичный IP отвечает `200`. Значит, в группе безопасности Yandex TCP 8000 открыт не для всех
  source-IP. Пользователю: открыть браузер; если не грузится — добавить в SG правило TCP 8000 со своего
  IP (или 0.0.0.0/0).

### Следующие шаги
- (опц.) В консоли Yandex открыть TCP 8000 в группе безопасности (для доступа из браузера).
- (опц.) `sudo systemctl enable docker` на ВМ — автостарт контейнера после перезагрузки.
- (опц.) Сменить простой API-ключ `88888888` в `.env` на ВМ.
- (опц.) Обновить `docs/DEPLOY.md` под итоговую процедуру (3.12 + CPU-torch + UTF-8).

---

## 2026-09-08 (продолжение 2) — Интеграция Giga-Embeddings-instruct-480M-0826 + GPU-реиндексация + GitHub

### Сделано (Done)
- Локальная модель `E:\Project\models\Giga-Embeddings-instruct-480M-0826` (Сбер, Qwen3Bidirectional,
  1024-dim, bf16, SentenceTransformer-формат с `trust_remote_code=True` → `modeling_gigarembed.py`)
  интегрирована в RAG.
- `config.py`: `model_dir` → `models/Giga-Embeddings-instruct-480M-0826`, `embedding_model_id` → имя
  модели, `embed_device="cuda"`, `embed_normalize=True`, добавлен `embed_trust_remote_code=True`.
  Офлайн-режим сохранён (`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`).
- `rag/engine.py`:
  - Новый класс `InstructGigaEmbeddings(Embeddings)`: грузит `SentenceTransformer` с `trust_remote_code`,
    `embed_documents` с `prompt_name="document"`, `embed_query` с `prompt_name="query"` (instruct-префикс
    только у query). `batch_size=32`, внешний лимит 4096 чанков/итерацию.
  - `build_embeddings()` выбирает `InstructGigaEmbeddings`, если в `config_sentence_transformers.json`
    задан `prompts.query`, иначе `HuggingFaceEmbeddings` с `trust_remote_code`.
  - PDF-загрузчик: `fitz` (PyMuPDF) с фолбэком на `pypdf` (ускорение ~30с против минут).
  - Порог иррелевантности `_MAX_IRRELEVANT_DISTANCE` поднят **0.60 → 1.35** (шкала дистанций Giga иная:
    REL 0.88..1.33, NOISE 1.37..1.65; порог 1.35 чисто разделяет).
- **Реиндексация на GPU (RTX 4060 8Gb)**: `chroma_db` пересобран → **40436 чанков**, коллекция
  `kodeksbot`. torch CUDA-сборка (`torch==2.11.0+cu128`), VRAM ~3.5 ГБ.
- Сервер перезапущен (`RAG_DISABLE_WATCHER=1`), модель на GPU; проверка `/ask`: зарплата/долг → 8
  источников, пельмени/соседи-шум → честно «нет ответа».
- `.gitignore` уже корректен (исключает `.env`, `chroma_db/`, `models/`, `data/docs/`, `*.log`).
- Создан приватный GitHub-репозиторий `kodeksbot`, сделан коммит, push в `main`.

### Грабли
- **torch CPU-only в venv**: ставил `torch==2.11.0+cu128` с индекса cu128 (cu124 не имеет колёс под
  Python 3.14.7). Без CUDA эмбеддинги падают/OOM.
- **Chroma cosine distance** = `1 - cosine` для нормализованных векторов → чем выше, тем хуже.
  Старый порог 0.60 (для rubert-tiny2) отсекал ВСЕ релевантные запросы новой модели.
- **BM25-вес** (kw_w=2.5 > vs_w=1.0) ловит морфологию, но «соседи шумят» даёт vector_dist=1.55
  (бытовой запрос далёк от сухих чанков кодексов) → правильно отсекается порогом.
- Токен GitHub (из Windows Credential Manager) случайно попал в вывод pwsh — рекомендую ротировать.

### Следующие шаги
- (опц.) Поднять качество: reranker (cross-encoder) поверх гибридной выдачи.
- (опц.) Извлечь корпус `data/docs/` в git-lfs / отдельное хранилище (сейчас в .gitignore).
- (опц.) Ротировать секреты, упомянутые в переписке (в т.ч. GitHub-токен).

---

## 2026-09-08 (продолжение) — UI-переписывание + диагностика качества поиска

### Сделано (Done)
- Переписан `static/index.html` (единый файл, vanilla JS/CSS, без сборщиков):
  - Тёмный «юридический» дизайн (золото/бордовый), адаптивный, индикатор статуса сервера (`/health`).
  - **Обязательное цитирование**: под ответом блок «📚 Источники (N)» с карточками
    (кодекс / стр. / фрагмент + «Копировать фрагмент»).
  - Drag&drop + кнопка загрузки (`/upload`, PDF/DOCX/TXT), модалка списка документов (`/documents`),
    кнопка переиндексации (`/ingest`).
  - История диалога (с источниками) в `localStorage`; XSS-safe рендер (`textContent`/`esc()`).
- Исправлен **баг стартовой инициализации**: `renderHistory()` падал на
  `appendChild($('emptyState'))` после `wrap.innerHTML=''` (элемент становился `null`).
  Теперь ссылка на `emptyEl` сохраняется. Добавлен глобальный `window.onerror` → видимая
  красная плашка с текстом ошибки внизу экрана.
- Сервер перезапущен (`C:\rag_project\venv\Scripts\python.exe -m uvicorn app:app --port 8000`,
  `RAG_DISABLE_WATCHER=1`); статику отдаёт с диска, новый UI виден сразу.
- **Диагностика `retrieve()`** (временный скрипт `_diag_retrieve.py`, read-only): индекс и ТК РФ в
  порядке. По юр. формулировке «заработная плата задержка выплата» ТК в топе 8/10; по разговорной
  «не платят зарплату, что делать» — мусор (АПК/ГК/КТМ/УПК/КоАП…). Корень: синоним
  «зарплата»≠«заработная плата» ломает BM25, а `rubert-tiny2` слаб на семантику.
  Выбран вариант «пока стоп» — код поиска/эмбеддинга не трогали.

### Следующие шаги (Next Steps) — приоритет
1. **Query expansion** (синонимы/переформулировка перед `retrieve`, либо несколько вариантов
   запроса + RRF-мерж). Быстро, индекс не пересобираем. ← следующая задача.
2. **Reranker (cross-encoder)** поверх гибридной выдачи (ориг. Next Steps №3).
3. **Смена эмбеддинга** `rubert-tiny2` → `ai-forever/sbert_large_nlu_ru` (ориг. №2): модель качать
   вручную (HF CDN заблок), пересборка ~41110 чанков.
4. (опц.) Вынести `data/docs/` в git-lfs / отдельное хранилище.
5. (опц.) Ротировать секреты (GigaChat/HF/GitHub), упомянутые в переписке.
6. Удалить временный `_diag_retrieve.py`.

### Грабли и находки (Gotchas)
- `renderHistory()` + `innerHTML=''` убивает `emptyState` в DOM → `getElementById` возвращает `null`.
  Храни ссылку на элемент до очистки.
- Вывод кириллицы в консоли `pwsh` — mojibake (UTF-8 как latin-1); данные корректны, сверяй байты.
- **RAG: разговорные/синонимичные запросы** (зарплата vs заработная плата) молча роняют BM25 +
  слабый tiny2 → мусор в топе → LLM честно «нет ответа», но мы показываем ложные источники.
  Лечится query expansion / сильным эмбеддингом.
- HF LFS CDN заблокирован в сети → модели качать вручную на другой машине.
- Запуск только через venv `C:\rag_project\venv\Scripts\python.exe` из `E:\Project`; watcher
  отключаем `RAG_DISABLE_WATCHER=1` при тестах UI.
- Проект `E:\Project` — git-репозиторий **без remote** → локальный коммит возможен, push — нет.

---

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
1. ~~Красивый веб-интерфейс~~ — **ЗАКРЫТО** (сессия 2026-09-08 продолжение: переписан `static/index.html`).
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
