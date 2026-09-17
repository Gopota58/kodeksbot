"""
FastAPI-обёртка над RAGEngine (КодексБот).

Сама логика RAG вынесена в `rag/engine.py`, здесь только HTTP-интерфейс:
аутентификация, rate limiting, CORS, отдача статики и маршрутизация на методы движка.
"""
import os
import time
import logging
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, Response
from pydantic import BaseModel

from config import settings
from rag.engine import RAGEngine

# --- Инициализация движка (загружает эмбеддинги и открывает векторную БД) ---
engine = RAGEngine(settings)

# Прогрев LLM при старте (холодный старт GigaChat/LM Studio).
try:
    engine.ask("Привет")
except Exception:
    pass

# Авто-переиндексация: следим за data/docs/ и при изменении пересобираем индекс.
if os.environ.get("RAG_DISABLE_WATCHER") != "1":
    engine.start_watcher()

# --- FastAPI приложение ---
app = FastAPI(
    title="КодексБот API",
    description="RAG-ассистент по российским кодексам (GigaChat + Chroma)",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials="*" not in settings.allowed_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# RATE LIMITING (in-memory, без внешних зависимостей)
# ============================================
# Публичное демо: один посетитель не должен выжечь квоту GigaChat. Ограничение —
# скользящее окно в 60 секунд на пару (IP, группа эндпоинтов). Состояние живёт
# в памяти процесса: для одного контейнера этого достаточно, для нескольких
# реплик понадобился бы Redis.
_rl_hits: dict[str, deque] = defaultdict(deque)
_rl_last_gc: float = 0.0
_RL_WINDOW = 60.0        # секунд
_RL_GC_EVERY = 300.0     # как часто подчищать протухшие записи


def _rate_limit_for(path: str) -> int | None:
    """Лимит (запросов в минуту) для пути; None — путь не ограничиваем."""
    if not settings.rate_limit_enabled:
        return None
    if path == "/ask":
        return settings.rate_limit_per_minute
    if path.startswith("/ingest") or path.startswith("/upload") or path.startswith("/documents"):
        return settings.rate_limit_admin_per_minute
    return None


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # CORS-preflight не считаем: он бесплатный (без обращения к LLM), а браузер
    # шлёт его перед каждым кросс-доменным POST — иначе лимит делился бы вдвое.
    if request.method == "OPTIONS":
        return await call_next(request)

    limit = _rate_limit_for(request.url.path)
    if limit is None:
        return await call_next(request)

    global _rl_last_gc
    now = time.monotonic()
    ip = request.client.host if request.client else "unknown"
    key = f"{ip}|{request.url.path}"

    hits = _rl_hits[key]
    while hits and now - hits[0] > _RL_WINDOW:
        hits.popleft()

    if len(hits) >= limit:
        retry_after = max(1, int(_RL_WINDOW - (now - hits[0])))
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Слишком много запросов с этого адреса. "
                          f"Повторите через {retry_after} с."
            },
            headers={"Retry-After": str(retry_after)},
        )

    hits.append(now)

    # Ленивая уборка, чтобы словарь не пух от разовых посетителей.
    if now - _rl_last_gc > _RL_GC_EVERY:
        for k in list(_rl_hits):
            dq = _rl_hits[k]
            while dq and now - dq[0] > _RL_WINDOW:
                dq.popleft()
            if not dq:
                del _rl_hits[k]
        _rl_last_gc = now

    return await call_next(request)


@app.on_event("startup")
def _enable_rag_engine_logging():
    rl = logging.getLogger("rag.engine")
    rl.disabled = False
    rl.propagate = True
    rl.setLevel(logging.INFO)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Аутентификация (API Key) ---
def verify_api_key(api_key: str = Header(None, alias="X-API-Key")):
    """Публичный ключ: только /ask. Он лежит в static/index.html и потому не секретен."""
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


def verify_admin_key(api_key: str = Header(None, alias="X-API-Key")):
    """Админский ключ: /upload, /ingest, /documents, DELETE /documents/{file}.

    В публичную статику не попадает — иначе любой посетитель демо смог бы
    удалить корпус или залить произвольный файл.
    """
    if not api_key or api_key != settings.resolved_admin_api_key:
        raise HTTPException(status_code=401, detail="Требуется админ-ключ (ADMIN_API_KEY)")
    return api_key


# --- Модели данных ---
class Question(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list = []


# ============================================
# ЭНДПОИНТЫ
# ============================================

@app.get("/", tags=["System"])
async def root():
    if os.path.exists(os.path.join("static", "index.html")):
        return RedirectResponse(url="/static/index.html")
    return {
        "message": "КодексБот API. Документация — /docs",
        "health": "/health",
        "ask": "POST /ask",
    }


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Отдаём SVG-иконку, чтобы браузер не получал 404 на каждый заход."""
    icon = os.path.join("static", "favicon.svg")
    if os.path.exists(icon):
        return RedirectResponse(url="/static/favicon.svg")
    return Response(status_code=204)


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask(question: Question, api_key: str = Depends(verify_api_key)):
    """Задать вопрос по кодексам (retrieval + генерация + источники)."""
    try:
        answer, sources = engine.ask_with_sources(question.question)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", tags=["Admin"])
async def ingest_documents(api_key: str = Depends(verify_admin_key)):
    """Принудительная переиндексация всех документов из data/docs/."""
    try:
        result = engine.reindex()
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", tags=["Admin"])
async def upload_file(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_admin_key),
):
    """Загрузить .pdf/.docx/.txt и переиндексировать все документы."""
    content = await file.read()
    try:
        result = engine.add_document(file.filename, content)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {
            "status": "ok",
            "message": f"Файл {file.filename} загружен и индексация выполнена",
            "details": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Admin"])
async def list_documents(api_key: str = Depends(verify_admin_key)):
    """Список документов в data/docs/."""
    return {"documents": engine.list_documents()}


@app.delete("/documents/{filename}", tags=["Admin"])
async def delete_document(
    filename: str,
    api_key: str = Depends(verify_admin_key),
):
    """Удалить документ и переиндексировать."""
    try:
        result = engine.remove_document(filename)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {
            "status": "ok",
            "message": f"Файл {filename} удалён, индексация выполнена",
            "details": result,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
