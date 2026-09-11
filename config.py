"""
Конфигурация проекта «КодексБот».

Все настройки вынесены в переменные окружения (.env) — больше нет
жёстко прописанных абсолютных путей и секретов, проект легко переносится.
"""
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

# --- Жёстко отключаем любую попытку скачивания моделей из сети ---
# Веса rubert-tiny2 уже лежат локально в models/rubert-tiny2 и должны загружаться
# только с диска. Если модель на диске отсутствует — код бросит явную ошибку,
# а не попытается что-то докачать. Это основная гарантия, что ни при первом
# запуске, ни когда-либо ещё не произойдёт обращения к Hugging Face за весами.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API-сервер ---
    api_key: str = "88888888"     # ключ доступа к /ask и админ-эндпоинтам (смените в проде!)
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Данные и векторная БД (пути относительны корня проекта) ---
    docs_dir: str = str(BASE_DIR / "data" / "docs")   # корпус: 26 кодексов РФ (PDF/DOCX)
    chroma_dir: str = str(BASE_DIR / "chroma_db")
    collection_name: str = "kodeksbot"
    model_dir: str = str(BASE_DIR / "models" / "Giga-Embeddings-instruct-480M-0826")
    embedding_model_id: str = "Giga-Embeddings-instruct-480M-0826"

    # --- Параметры эмбеддингов ---
    embed_device: str = os.getenv("EMBED_DEVICE", "cuda")  # CPU в облаке: задайте EMBED_DEVICE=cpu
    embed_normalize: bool = True
    embed_trust_remote_code: bool = True  # Giga-модель требует кастомный код Сбера

    # --- Провайдер эмбеддингов ---
    # "local" — HuggingFace (cointegrated/rubert-tiny2), грузится СТРОГО локально
    #            из models/rubert-tiny2. Скачивание из сети отключено (см. HF_HUB_OFFLINE).
    # "api"   — OpenAI-совместимый endpoint (напр. nomic-embed-text в LM Studio).
    embed_provider: str = "local"
    embed_api_base_url: str = ""      # пусто -> берётся llm_base_url
    embed_api_model: str = "text-embedding-nomic-embed-text-v1.5"
    embed_api_key: str = "lm-studio"

    # --- Ретривер ---
    retriever_k: int = 8
    enable_hyde: bool = True    # HyDE: генерить гипотетический фрагмент кодекса для улучшения поиска
    # Reranker: путь к локальной sentence-transformer модели (напр. all-MiniLM-L6-v2) для
    # переранжирования кандидатов гибридного поиска. Пустая строка = reranker отключён.
    rerank_model: str = str(BASE_DIR / "models" / "all-MiniLM-L6-v2")

    # --- LLM (GigaChat — Сбер, российский облачный LLM) ---
    llm_provider: str = "gigachat"   # "local" (LM Studio/Ollama), "openai" или "gigachat"
    llm_base_url: str = ""            # для gigachat не используется (берётся gigachat_base_url)
    llm_api_key: str = ""             # для gigachat — Authorization key из developers.sber.ru
    llm_model: str = "GigaChat-2"     # см. GET /v1/models (GigaChat-2 / GigaChat-Pro / GigaChat-Max)
    llm_temperature: float = 0.0
    llm_top_p: float = 0.9
    llm_max_tokens: int = 1024
    llm_use_system_prompt: bool = True
    llm_max_context_chars: int = 6000
    llm_extra_body: str = "{}"

    # --- GigaChat (Сбер) ---
    gigachat_base_url: str = "https://api.giga.chat/v1"
    gigachat_verify_ssl_certs: bool = True
    gigachat_ca_bundle_file: str = ""   # Russian Trusted Root CA (gosuslugi.ru/crt) — нужен на Windows

    # --- CORS (через запятую; "*" — разрешить все) ---
    allowed_origins: str = "*"

    # --- Telegram-бот (опц.) ---
    telegram_bot_token: str = ""
    telegram_proxy: str = ""
    rag_api_url: str = "http://localhost:8000"

    @property
    def allowed_origins_list(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    def resolve_embedding_model(self) -> str:
        """Путь к локальной модели эмбеддингов (models/rubert-tiny2).

        Всегда возвращает локальную папку. Никогда не возвращает HF repo-id,
        поэтому скачивание из сети исключено: при отсутствии весов на диске
        бросается явная ошибка (см. HF_HUB_OFFLINE в начале файла).
        """
        if getattr(self, "embed_provider", "local") != "local":
            return self.embedding_model_id
        local = Path(self.model_dir)
        if not (local.exists() and (local / "config.json").exists()):
            raise FileNotFoundError(
                f"Локальная модель эмбеддингов не найдена в: {local}\n"
                f"Положите веса rubert-tiny2 (model.safetensors, pytorch_model.bin, "
                f"tinybert-ru-labse-adapter-v2.pt) в эту папку. Авто-загрузка из "
                f"сети отключена (HF_HUB_OFFLINE=1)."
            )
        return str(local)


settings = Settings()
