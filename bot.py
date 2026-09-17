"""
Telegram-бот «КодексБот» ⚖️

Тонкий HTTP-клиент к локальному RAG-серверу (app.py на :8000). Бот НЕ держит
свой экземпляр RAGEngine — единственный владелец векторного индекса и
авто-переиндексации (watcher) — веб-сервер. Это исключает гонку двух процессов
за одну chroma_db и «протухание» кэша коллекции.

Поток: сообщение в Telegram -> POST /ask на веб-сервер -> ответ в Telegram.

ОБХОД БЛОКИРОВКИ TELEGRAM В РФ
------------------------------
В России api.telegram.org заблокирован на уровне сети (прямой доступ — таймаут).
Если на машине поднят локальный SOCKS5/HTTP-прокси, весь трафик бота к Telegram
идёт ЧЕРЕЗ него (поле proxy в HTTPXRequest). К локальному веб-серверу (:8000)
бот ходит напрямую (trust_env=False), прокси не трогая.

Настройка через .env:
  TELEGRAM_BOT_TOKEN — токен бота
  TELEGRAM_PROXY     — прокси к Telegram (http://127.0.0.1:7890 или socks5://...)
  RAG_API_URL        — адрес RAG-сервера (по умолчанию http://localhost:8000)
  API_KEY            — публичный ключ X-API-Key для /ask (берётся из config)
  ADMIN_API_KEY      — админский ключ для /ingest (пусто = используется API_KEY)
"""
import asyncio
import logging

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from config import settings

# --- Конфигурация: все секреты только из .env (через pydantic-settings) ---
TOKEN = settings.telegram_bot_token
# Прокси для обхода блокировки Telegram. Пустая строка = прямое соединение.
PROXY = settings.telegram_proxy
# Локальный RAG-сервер (веб-приложение app.py).
RAG_API_URL = settings.rag_api_url
RAG_API_KEY = settings.api_key                        # публичный: только /ask
RAG_ADMIN_KEY = settings.resolved_admin_api_key       # админ: /ingest

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("kodeksbot_bot")

# HTTP-клиенты к локальному RAG-серверу. trust_env=False — ходить на localhost
# напрямую, минуя любые переменные прокси (иначе запрос уйдёт в Telegram-прокси).
_rag = httpx.Client(
    base_url=RAG_API_URL,
    headers={"X-API-Key": RAG_API_KEY},
    timeout=httpx.Timeout(120.0),
    trust_env=False,
)
# Отдельный клиент для админ-эндпоинтов: /ingest с публичным ключом вернёт 401.
_rag_admin = httpx.Client(
    base_url=RAG_API_URL,
    headers={"X-API-Key": RAG_ADMIN_KEY},
    timeout=httpx.Timeout(600.0),   # переиндексация корпуса — долгая операция
    trust_env=False,
)


def ask_rag(question: str) -> str:
    """Синхронный запрос к /ask веб-сервера."""
    r = _rag.post("/ask", json={"question": question})
    r.raise_for_status()
    return r.json().get("answer", "")


def reindex_rag() -> dict:
    """Синхронный вызов /ingest (полная переиндексация docs/) по админ-ключу."""
    r = _rag_admin.post("/ingest")
    r.raise_for_status()
    return r.json()


def split_text(text: str, limit: int = 4000):
    """Дробит длинный ответ на части по <=limit символов (лимит Telegram — 4096)."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 > limit:
            if cur:
                parts.append(cur)
            cur = para
            while len(cur) > limit:
                parts.append(cur[:limit])
                cur = cur[limit:]
        else:
            cur = (cur + "\n" + para) if cur else para
    if cur:
        parts.append(cur)
    return parts


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я КодексБот ⚖️ — ассистент по российскому законодательству.\n"
        "Задай вопрос по кодексам РФ — отвечу со ссылками на конкретные статьи "
        "и фрагменты документов.\n\nКоманды: /start, /help, /reindex"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Просто задавай юридический вопрос, например:\n"
        "• каков срок исковой давности по гражданским делам?\n"
        "• какая ответственность за задержку зарплаты?\n"
        "• что считается мелким хищением?\n\n"
        "Я ищу ответ в корпусе кодексов и отвечаю по делу, указывая источник.\n"
        "/reindex — пересобрать индекс вручную (обычно не нужен: "
        "папка data/docs/ переиндексируется автоматически)."
    )


async def reindex_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        res = await asyncio.to_thread(reindex_rag)
        msg = res.get("message") or res.get("status") or "готово"
        await update.message.reply_text(f"🔄 Переиндексация: {msg}")
    except httpx.HTTPStatusError as e:
        await update.message.reply_text(
            f"⚠️ Ошибка переиндексации: HTTP {e.response.status_code}"
        )
    except Exception:
        log.exception("Ошибка /reindex")
        await update.message.reply_text(
            "⚠️ RAG-сервер недоступен. Проверь, запущен ли веб-сервер (app.py)."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    question = update.message.text.strip()
    if not question:
        return

    chat_id = update.effective_chat.id
    # Показываем «печатает…», пока сервер думает
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        # ask_rag — синхронный HTTP-вызов (может ждать LLM несколько секунд):
        # выносим в поток, чтобы не блокировать event loop.
        answer = await asyncio.to_thread(ask_rag, question)
    except httpx.ConnectError:
        log.error("RAG-сервер недоступен (%s)", RAG_API_URL)
        answer = "⚠️ Сервер базы знаний сейчас недоступен. Запусти веб-сервер (app.py)."
    except Exception:
        log.exception("Ошибка /ask при вопросе: %r", question)
        answer = "Извини, сейчас не могу найти ответ (ошибка сервера). Попробуй позже."

    for part in split_text(answer):
        await update.message.reply_text(part)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.warning("Ошибка при обновлении %s: %s", update, context.error)


def main():
    if not TOKEN:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN не задан. Скопируйте .env.example в .env и укажите "
            "токен от @BotFather (или экспортируйте переменную окружения)."
        )
    builder = ApplicationBuilder().token(TOKEN)
    if PROXY:
        log.info("Использую прокси для Telegram: %s", PROXY)
        # trust_env=False — чтобы системные HTTPS_PROXY/ALL_PROXY не конфликтовали
        # с явно заданным обходным прокси.
        builder = builder.request(
            HTTPXRequest(
                proxy=PROXY,
                connect_timeout=20,
                read_timeout=60,
                httpx_kwargs={"trust_env": False},
            )
        )
    else:
        log.info("Прокси не задан — прямое соединение с Telegram.")

    app = builder.build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reindex", reindex_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    log.info("Бот запущен (long polling)… RAG-сервер: %s", RAG_API_URL)
    # bootstrap_retries — повторять попытки подключения к Telegram при
    # транзиентных сбоях сети/прокси при старте (а не падать сразу).
    app.run_polling(close_loop=False, bootstrap_retries=10)


def run_supervised():
    """Супервизор: держит бота запущенным. Если run_polling завершился
    (непредвиденный выход/падение), делает паузу и перезапускает цикл,
    чтобы бот не «отваливался» при кратковременных обрывах прокси/VPN."""
    while True:
        try:
            main()
        except KeyboardInterrupt:
            log.info("Получен сигнал остановки. Выход.")
            break
        except Exception:
            log.exception("Бот неожиданно остановился. Перезапуск через 5 с…")
            import time

            time.sleep(5)


if __name__ == "__main__":
    run_supervised()
