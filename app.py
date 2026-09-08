"""
FastAPI-обёртка над RAGEngine (КодексБот).

Сама логика RAG вынесена в `rag/engine.py`, здесь только HTTP-интерфейс:
аутентификация, CORS, отдача статики и маршрутизация на методы движка.
"""
import os
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
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
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
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


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask(question: Question, api_key: str = Depends(verify_api_key)):
    """Задать вопрос по кодексам (retrieval + генерация + источники)."""
    try:
        answer, sources = engine.ask_with_sources(question.question)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", tags=["Admin"])
async def ingest_documents(api_key: str = Depends(verify_api_key)):
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
    api_key: str = Depends(verify_api_key),
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
async def list_documents(api_key: str = Depends(verify_api_key)):
    """Список документов в data/docs/."""
    return {"documents": engine.list_documents()}


@app.delete("/documents/{filename}", tags=["Admin"])
async def delete_document(
    filename: str,
    api_key: str = Depends(verify_api_key),
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
