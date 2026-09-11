"""
RAGEngine — переиспользуемое ядро движка RAG (адаптировано под КодексБот).

Инкапсулирует всю логику: эмбеддинги, векторную БД (Chroma), гибридный ретривер
(вектор + BM25/TF-IDF, слияние через Reciprocal Rank Fusion), LLM (GigaChat) и
RAG-цепочку. Загрузка документов — PDF / DOCX / TXT (корпус российских кодексов).
"""
import os
import json
import pathlib
import re
import threading
import httpx
import logging

import chromadb
import chardet
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from config import settings as default_settings

try:
    from sentence_transformers import SentenceTransformer as _STransformer
    _HAVE_ST = True
except Exception:  # sentence-transformers может отсутствовать — reranker тогда отключается
    _HAVE_ST = False
    _STransformer = None

log = logging.getLogger(__name__)


# --- Промпт ассистента (юридический, СТРОГО по контексту, обязательное цитирование) ---
CANONICAL_NO_ANSWER = "В предоставленных документах нет ответа на этот вопрос."

SYSTEM_INSTRUCTION = """Ты — КодексБот, юридический ассистент по российским кодексам и законам.

ЖЁСТКИЕ ПРАВИЛА (нарушение недопустимо):
1. Отвечай СТРОГО и ИСКЛЮЧИТЕЛЬНО на основе Контекста (цитат из кодексов), данного ниже. НЕ ИСПОЛЬЗУЙ никакие факты, законы, статьи или ФЗ, которых НЕТ в Контексте.
2. Если в Контексте нет ответа на вопрос, ответь ТОЛЬКО этой фразой и ничего больше не добавляй:
   «В предоставленных документах нет ответа на этот вопрос.»
3. НИКОГДА не придумывай и не добавляй ссылки на законы/статьи/ФЗ вне Контекста (например, «согласно ФЗ-400», «по ст. 8»), если их нет в Контексте.
4. Всегда отвечай по-русски, кратко и по существу, без вступлений.
5. Обязательно указывай источник из Контекста: название документа и, при наличии, номер статьи/страницу. Если релевантных статей несколько — перечисли их.
 6. Если Контекст лишь упоминает тему вопроса, но НЕ содержит точного ответа (конкретных сроков, возраста, сумм, условий применения), — отвечай ТОЛЬКО фразой «нет ответа» и не добавляй ничего от себя."""

# Few-shot: задаёт правильное поведение (отвечать ТОЛЬКО из контекста + цитировать статью;
# при отсутствии контекста — только каноническая фраза «нет ответа»).
_FEWSHOT = [
    ("human", "Контекст:\nСТАТЬЯ 81. РАСТОРЖЕНИЕ ТРУДОВОГО ДОГОВОРА РАБОТОДАТЕЛЕМ. 1) неоднократное неисполнение работником без уважительных причин трудовых обязанностей; 6) однократное грубое нарушение работником трудовых обязанностей (прогул, появление в состоянии опьянения).\n\nВопрос: за что могут уволить?"),
    ("ai", "По Трудовому кодексу РФ работодатель вправе уволить по ст. 81: неоднократное неисполнение обязанностей, прогул, появление в состоянии опьянения, а также при сокращении штата. Источник: Трудовой кодекс РФ, ст. 81."),
    ("human", "Контекст:\n(пусто)\n\nВопрос: с какого возраста уходят на пенсию?"),
    ("ai", CANONICAL_NO_ANSWER),
    ("human", "Контекст:\nСТАТЬЯ 85. ПРАВО НА ПОЛУЧЕНИЕ АЛИМЕНТОВ. 2. Алименты на детей уплачиваются до достижения ими совершеннолетия, то есть до 18 лет.\n\nВопрос: до какого возраста уплачиваются алименты?"),
    ("ai", "Алименты на детей уплачиваются до достижения ими совершеннолетия — то есть до 18 лет (ст. 85 Семейного кодекса РФ). Источник: Семейный кодекс РФ, ст. 85."),
    ("human", "Контекст:\nСТАТЬЯ 15. Средства ... пенсионное обеспечение работников (в Трудовом кодексе лишь общие упоминания пенсий, без указания возраста выхода на пенсию по старости).\n\nВопрос: в каком возрасте выходят на пенсию по старости в России?"),
    ("ai", CANONICAL_NO_ANSWER),
]


def _build_prompt(use_system: bool) -> ChatPromptTemplate:
    if use_system:
        return ChatPromptTemplate.from_messages(
            [("system", SYSTEM_INSTRUCTION)] + list(_FEWSHOT) +
            [("human", "Контекст:\n{context}\n\nВопрос: {input}")]
        )
    return ChatPromptTemplate.from_messages(
        list(_FEWSHOT) +
        [("human", SYSTEM_INSTRUCTION + "\n\nКонтекст:\n{context}\n\nВопрос: {input}")]
    )


# --- Поиск по номеру статьи + нормализация запроса (помогает «понимать» вопросы) ---
# Заголовок статьи в тексте чанка: «СТАТЬЯ 105» / «Статья 81»
ART_HEADER_RE = re.compile(r"СТАТЬЯ\s*(\d+)", re.IGNORECASE)
# Упоминание номера статьи в вопросе пользователя
ART_QUERY_RE = re.compile(r"(?:стать[яею]|ст\.?)\s*(\d+)", re.IGNORECASE)
ART_QUERY_RE2 = re.compile(r"(\d+)\s*ст", re.IGNORECASE)
# Ключевое слово кодекса в вопросе -> подстрока имени файла (для фильтрации статьи).
# Только длинные, однозначные слова, чтобы не ловить подстроки в словах-паразитах
# (напр. «ск» ловило бы «скажи»).
CODEX_HINTS = [
    ("уголовн", "Уголовный_кодекс"),
    ("трудов", "Трудовой_кодекс"),
    ("семейн", "Семейный_кодекс"),
    ("граждан", "Гражданский_кодекс"),
    ("налог", "Налоговый_кодекс"),
    ("административн", "Кодекс_об_административных_правонарушениях"),
    ("жилищн", "Жилищный_кодекс"),
    ("бюджетн", "Бюджетный_кодекс"),
    ("земельн", "Земельный_кодекс"),
    ("лесн", "Лесной_кодекс"),
    ("водн", "Водный_кодекс"),
    ("воздушн", "Воздушный_кодекс"),
    ("арбитражн", "Арбитражный_процессуальный_кодекс"),
    ("уголовно-процессуальн", "Уголовно-процессуальный_кодекс"),
    ("гражданско-процессуальн", "Гражданский_процессуальный_кодекс"),
    ("уголовно-исполнительн", "Уголовно-исполнительный_кодекс"),
    ("таможенн", "Таможенный_кодекс"),
    ("конституц", "Конституция"),
]
# Слова-паразиты в вопросах — убираем перед поиском (не несут юридического смысла)
_FILLER_RE = re.compile(
    r"\b(скажи|пожалуйста|помоги|подскажи|объясни|у меня|вопрос|какой закон|где написано|"
    r"что делать если|как мне|мне нужно|интересно|можно ли|нужно ли|есть ли|правильно ли|"
    r"законно ли|я хочу|хотел бы|расскажи|что говорит|как узнать|как правильно|подскажите|"
    r"уточни|проконсультируй|совет)\b", re.IGNORECASE)


# --- Авто-определение кодировки (.txt могут быть в UTF-8 или Windows-1251) ---
def detect_encoding(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read(10000)
    return (chardet.detect(raw) or {}).get("encoding") or "utf-8"


class AutoDetectTextLoader:
    """Обёртка над TextLoader с авто-детектом кодировки."""

    def __init__(self, file_path, **kwargs):
        from langchain_community.document_loaders import TextLoader
        encoding = detect_encoding(file_path)
        self._inner = TextLoader(file_path, encoding=encoding, **kwargs)

    def load(self):
        return self._inner.load()


# --- Загрузка документов: PDF / DOCX / TXT ---
def _load_file(path: str) -> list:
    p = pathlib.Path(path)
    ext = p.suffix.lower()
    if ext == ".txt":
        return AutoDetectTextLoader(str(p)).load()
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF — значительно быстрее pypdf и не зависает на проблемных PDF
            doc = fitz.open(str(p))
            docs = []
            for i, page in enumerate(doc, 1):
                txt = page.get_text() or ""
                if txt.strip():
                    # метаданные позволяют цитировать «документ + страница»
                    docs.append(Document(page_content=txt, metadata={"source": p.name, "page": i}))
            doc.close()
            return docs
        except Exception:
            from pypdf import PdfReader
            reader = PdfReader(str(p))
            docs = []
            for i, page in enumerate(reader.pages, 1):
                txt = page.extract_text() or ""
                if txt.strip():
                    # метаданные позволяют цитировать «документ + страница»
                    docs.append(Document(page_content=txt, metadata={"source": p.name, "page": i}))
            return docs
    if ext == ".docx":
        import docx
        document = docx.Document(str(p))
        text = "\n".join(par.text for par in document.paragraphs if par.text.strip())
        return [Document(page_content=text, metadata={"source": p.name})]
    log.warning("Неподдерживаемый формат пропущен: %s", p.name)
    return []


def load_all_documents(docs_dir: str) -> list:
    docs = []
    if not os.path.isdir(docs_dir):
        return docs
    for f in sorted(os.listdir(docs_dir)):
        if f.lower().endswith((".txt", ".pdf", ".docx")):
            try:
                docs.extend(_load_file(os.path.join(docs_dir, f)))
            except Exception as e:
                log.warning("Не удалось загрузить %s: %s", f, e)
    return docs


class HashingEmbeddings(Embeddings):
    """Детерминированный эмбеддер без внешних моделей и сети: feature hashing
    символьных триграмм. Нужен только для ГЕРМЕТИЧНЫХ тестов/CI."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        import hashlib
        v = np.zeros(self.dim, dtype=np.float32)
        t = text.lower()
        for i in range(max(len(t) - 2, 0)):
            g = t[i:i + 3].encode("utf-8")
            idx = int.from_bytes(hashlib.blake2b(g, digest_size=4).digest(), "big") % self.dim
            v[idx] += 1.0
        n = float(np.linalg.norm(v))
        return (v / n).tolist() if n else v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class InstructGigaEmbeddings(Embeddings):
    """Обёртка над SentenceTransformer для instruct-эмбеддинг-моделей (Сбер Giga).

    SentenceTransformer хранит инструкции в config_sentence_transformers.json
    (prompts.query / prompts.document). При encode с prompt_name подставляется
    нужный префикс: для query — «Instruct: Given a query, retrieve relevant
    passages\nQuery: », для документов — пусто. Обычный HuggingFaceEmbeddings не
    применяет prompt_name дифференцированно, поэтому для instruct-моделей нужна
    эта обёртка. Кастомный код Сбера (modeling_gigarembed.py) требует
    trust_remote_code=True.
    """

    def __init__(self, model_path, device="cpu", normalize=True, trust_remote_code=True):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_path,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs={"torch_dtype": "auto"},
        )
        self.normalize = normalize

    def embed_documents(self, texts):
        # Батчируем сами, чтобы видеть прогресс. batch_size ограничен VRAM RTX 4060
        # (8 Gb): крупные батчи вызывают CUDA OOM из-за квадратичных attention-матриц.
        outer = 4096
        out = []
        n = len(texts)
        for i in range(0, n, outer):
            chunk = texts[i:i + outer]
            emb = self.model.encode(
                chunk,
                prompt_name="document",
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            )
            out.append(np.asarray(emb, dtype=np.float32))
            print(f"[embed] {min(i + outer, n)}/{n} docs", flush=True)
        return np.concatenate(out).tolist() if out else []

    def embed_query(self, text):
        emb = self.model.encode(
            [text],
            prompt_name="query",
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb[0], dtype=np.float32).tolist()


# --- Фабрики ---
def _load_local_embeddings(s):
    """Локальные эмбеддинги (sentence-transformers/torch), строго с диска.

    Обычные модели (rubert-tiny2) — через HuggingFaceEmbeddings. Instruct-модели
    (Giga-Embeddings-instruct-*) — через InstructGigaEmbeddings (подставляет
    query/document-промпты, обязателен trust_remote_code для кастомного кода Сбера).
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    local_model = s.resolve_embedding_model()
    trust = getattr(s, "embed_trust_remote_code", False)

    # Детект instruct-режима: config_sentence_transformers.json с непустым query-промптом
    st_cfg = pathlib.Path(local_model) / "config_sentence_transformers.json"
    is_instruct = False
    if st_cfg.exists():
        try:
            data = json.loads(st_cfg.read_text(encoding="utf-8"))
            prompts = (data or {}).get("prompts") or {}
            is_instruct = bool(prompts.get("query"))
        except Exception:
            is_instruct = False

    if is_instruct:
        return InstructGigaEmbeddings(
            local_model, device=s.embed_device, normalize=s.embed_normalize,
            trust_remote_code=trust,
        )
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=local_model,
        model_kwargs={"device": s.embed_device, "trust_remote_code": trust},
        encode_kwargs={"normalize_embeddings": s.embed_normalize},
        cache_folder=str(pathlib.Path(local_model).parent),
    )


def build_embeddings(s=None):
    s = s or default_settings
    if getattr(s, "embed_provider", "local") == "api":
        from langchain_openai import OpenAIEmbeddings
        base = s.embed_api_base_url or s.llm_base_url
        kwargs = dict(model=s.embed_api_model, base_url=base, api_key=s.embed_api_key,
                      check_embedding_ctx_length=False)
        if base and ("localhost" in base or "127.0.0.1" in base):
            kwargs["http_client"] = httpx.Client(trust_env=False)
        return OpenAIEmbeddings(**kwargs)
    if getattr(s, "embed_provider", "local") == "hash":
        return HashingEmbeddings()
    # Локальные эмбеддинги (Giga / rubert-tiny2 через sentence-transformers/torch).
    # Грузим СТРОГО из локальной папки, без обращения в сеть. HF_HUB_OFFLINE=1
    # уже выставлен в config.py — дублируем здесь как страховку.
    return _load_local_embeddings(s)


def _parse_extra_body(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_llm(s=None):
    s = s or default_settings
    # GigaChat (Сбер) — российский облачный LLM. SDK сам меняет Authorization key
    # (LLM_API_KEY) на access_token через OAuth и обновляет его.
    if s.llm_provider == "gigachat":
        if not s.llm_model:
            raise ValueError("Для провайдера gigachat задайте LLM_MODEL в .env")
        from langchain_gigachat import GigaChat
        kwargs = dict(
            model=s.llm_model,
            credentials=s.llm_api_key,
            temperature=s.llm_temperature,
            top_p=s.llm_top_p,
            max_tokens=s.llm_max_tokens,
            base_url=s.gigachat_base_url,
            verify_ssl_certs=s.gigachat_verify_ssl_certs,
        )
        if s.gigachat_ca_bundle_file:
            kwargs["ca_bundle_file"] = s.gigachat_ca_bundle_file
        return GigaChat(**kwargs)
    extra = _parse_extra_body(s.llm_extra_body)
    extra["max_tokens"] = s.llm_max_tokens
    kwargs = dict(model=s.llm_model, temperature=s.llm_temperature, top_p=s.llm_top_p,
                  api_key=s.llm_api_key, extra_body=extra)
    if s.llm_base_url:
        kwargs["base_url"] = s.llm_base_url
    if s.llm_base_url and ("localhost" in s.llm_base_url or "127.0.0.1" in s.llm_base_url):
        kwargs["http_client"] = httpx.Client(
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
            timeout=httpx.Timeout(90.0),
        )
    return ChatOpenAI(**kwargs)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


class RAGEngine:
    def __init__(self, settings=default_settings, embeddings=None, llm=None):
        self.settings = settings
        self.embeddings = embeddings or build_embeddings(self.settings)
        self.llm = llm or build_llm(self.settings)
        # Reranker (опц., config.rerank_model): локальная sentence-transformer модель
        # (напр. all-MiniLM-L6-v2), переранжирует кандидатов гибридного поиска по
        # семантической близости запросу. Работает офлайн (модель уже в models/).
        # При сбое/отсутствии — self._rerank_model=None → reranker молча отключается.
        self._rerank_model = None
        rm = getattr(self.settings, "rerank_model", "") or ""
        if _HAVE_ST and rm:
            try:
                self._rerank_model = _STransformer(str(rm), device=self.settings.embed_device)
                log.info("Reranker загружен: %s", rm)
            except Exception as e:
                log.warning("Reranker не загрузился (%s), отключаем: %s", rm, e)
                self._rerank_model = None
        self.collection_name = self.settings.collection_name
        self.client = chromadb.PersistentClient(path=self.settings.chroma_dir)
        self._lock = threading.Lock()
        self._watcher_running = False
        self._query_cache = {}  # кэш вариантов запроса: вопрос -> [варианты]
        self._build()

    def _build_chain(self):
        prompt = _build_prompt(self.settings.llm_use_system_prompt)

        def _ctx(q):
            ctx = format_docs(self.retrieve(q))
            limit = self.settings.llm_max_context_chars
            if limit and len(ctx) > limit:
                ctx = ctx[:limit]
                if "\n" in ctx:
                    ctx = ctx[: ctx.rfind("\n")]
                ctx = ctx + "\n…(контекст усечён по лимиту)"
            return ctx

        self._prompt = prompt
        return (
            {"context": _ctx, "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _build_keyword_index(self):
        try:
            coll = self.client.get_collection(self.collection_name)
            texts, metas = [], []
            offset = 0
            while True:
                # пагинация: прямой get() по всей коллекции (41k+ чанков)
                # упирается в лимит переменных SQLite, поэтому читаем порциями
                data = coll.get(include=["documents", "metadatas"], limit=5000, offset=offset)
                docs = data.get("documents") or []
                if not docs:
                    break
                texts.extend(docs)
                metas.extend(data.get("metadatas") or [])
                if len(docs) < 5000:
                    break
                offset += 5000
        except Exception:
            texts, metas = [], []
        self._chunk_texts = texts
        self._chunk_metadatas = metas
        if texts:
            self._vectorizer = TfidfVectorizer()
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        else:
            self._vectorizer = None

    def _build(self):
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.settings.retriever_k}
        )
        self._build_keyword_index()
        self.rag_chain = self._build_chain()

    # --- Гибридный ретривер: вектор + BM25, слияние через Reciprocal Rank Fusion ---
    # --- Раскрытие синонимов/переформулировка для русского юр-корпуса ---
    # Двунаправленные группы синонимов: если в запросе встречается любой «триггер»,
    # к запросу добавляются «расширения» (обратная связь работает, т.к. триггерами
    # выступают и разговорные, и юридические формы одного понятия, напр.
    # зарплата <-> заработная плата).
    _SYNONYM_GROUPS = [
        (["зарплат", "заработн"], ["заработная плата", "оплата труда", "выплата заработной платы"]),
        (["оплат труда"], ["заработная плата", "зарплата", "выплата заработной платы"]),
        (["неустойк"], ["пени", "штраф", "штрафные санкции"]),
        (["пени"], ["неустойка", "штраф"]),
        (["штраф"], ["неустойка", "пени", "административный штраф"]),
        (["увольн", "увол"], ["расторжение трудового договора", "сокращение численности"]),
        (["сокращ"], ["увольнение", "расторжение трудового договора"]),
        (["декрет", "беременн"], ["отпуск по беременности и родам"]),
        (["больничн"], ["лист нетрудоспособности", "пособие по временной нетрудоспособности"]),
        (["ипотек"], ["залог недвижимости", "жилищный кредит", "ипотечный кредит"]),
        (["алимент"], ["содержание ребёнка", "содержание детей"]),
        (["наслед"], ["завещание", "наследник", "вступление в наследство"]),
        (["жил", "квартир"], ["жилое помещение", "жилищный кодекс"]),
        (["аренд"], ["договор аренды", "наём имущества"]),
        (["налог"], ["налог", "налоговый кодекс", "недоимка", "налоговая декларация"]),
        (["ндфл", "возврат налог"], ["налоговый вычет", "возврат налога", "налоговый агент", "возврат ндфл"]),
        (["трудов"], ["трудовой договор", "трудовые отношения"]),
        (["развод", "развест", "брак расторг"], ["расторжение брака", "семейный кодекс"]),
        (["договор"], ["соглашение", "контракт"]),
        (["возмещ"], ["возмещение ущерба", "компенсация"]),
        (["убыт"], ["убытки", "возмещение убытков"]),
        (["иск"], ["исковое заявление", "судебный иск"]),
        (["суд"], ["судебное разбирательство", "исковое заявление", "подсудность", "рассмотрение споров", "арбитражный суд"]),
        (["подсудн", "подсуд"], ["подсудность", "территориальная подсудность", "рассмотрение споров"]),
        (["юрисдикц"], ["подсудность", "судебная юрисдикция", "рассмотрение споров"]),
        (["рассмотр"], ["рассмотрение дела", "подсудность", "исковое заявление"]),
        (["защит"], ["защита прав", "охрана прав"]),
        (["лиценз"], ["лицензия", "разрешение"]),
        (["административн"], ["административное правонарушение", "административный штраф"]),
        (["уголовн"], ["преступление", "уголовная ответственность"]),
        (["собственн"], ["собственность", "право собственности", "имущество"]),
        (["расторг"], ["расторжение", "прекращение договора"]),
        (["недействител"], ["недействительность", "ничтожная сделка", "оспоримая сделка"]),
        (["ответственн"], ["ответственность", "штраф", "наказание", "взыскание"]),
        # --- доп. покрытие разговорных/смежных формулировок (D) ---
        (["прогул"], ["увольнение за прогул", "статья 81", "расторжение трудового договора"]),
        (["брачн"], ["брачный договор", "договор брачного договора"]),
        (["материнск"], ["материнский капитал"]),
        (["раздел", "делёж", "дележ"], ["раздел совместно нажитого имущества", "раздел имущества супругов"]),
        (["жкх", "коммунал"], ["жилищно-коммунальное хозяйство", "коммунальные услуги"]),
        (["пенсионн", "пенс"], ["пенсия", "страховая пенсия"]),
        (["усыновл", "удочер"], ["усыновление", "приёмная семья"]),
        # --- налоги: неуплата / недоимка / задолженность (покрытие НК РФ) ---
        (["неуплат", "неуплата"], ["недоимка", "задолженность по налогам", "взыскание налога", "налоговая задолженность"]),
        (["недоимк"], ["неуплата налогов", "задолженность по налогам", "взыскание налога"]),
        (["задолженн", "задолжен"], ["недоимка", "неуплата налогов", "взыскание налога"]),
    ]

    # (фраза_в_ответе, метка_файла) — используется grounding-санитайзером.
    # Метка_файла должна встречаться в имени файла документа из data/docs/
    # (напр. «Налоговый_кодекс_РФ_часть_1.docx» → «налог»).
    _CODEX_ALIASES = [
        ("трудовой кодекс", "трудов"),
        ("налоговый кодекс", "налог"),
        ("уголовный кодекс", "уголовн"),
        ("семейный кодекс", "семейн"),
        ("гражданский кодекс", "граждан"),
        ("жилищный кодекс", "жилищн"),
        ("конституция рф", "конституц"),
        ("бюджетный кодекс", "бюджетн"),
        ("земельный кодекс", "земельн"),
        ("лесной кодекс", "лесн"),
        ("водный кодекс", "водн"),
        ("воздушный кодекс", "воздушн"),
        ("таможенный кодекс", "таможенн"),
        ("уголовно-исполнительный кодекс", "уголовно-исполнительн"),
        ("уголовно-процессуальный кодекс", "уголовно-процессуальн"),
        ("гражданский процессуальный кодекс", "гражданско-процессуальн"),
        ("арбитражный процессуальный кодекс", "арбитражн"),
        ("кодекс об административных правонарушениях", "коап"),
        ("коап", "коап"),
        # аббревиатуры в ответе
        ("нк рф", "налог"), ("нк рф", "налог"),
        ("тк рф", "трудов"), ("ук рф", "уголовн"), ("ск рф", "семейн"),
        ("гк рф", "граждан"), ("жк рф", "жилищн"), ("бк рф", "бюджетн"),
        ("упк", "уголовно-процессуальн"), ("гпк", "гражданско-процессуальн"),
        ("апк", "арбитражн"), ("крф", "конституц"),
    ]

    def _expand_query(self, question: str) -> str:
        """Добавляет к запросу юридические синонимы из _SYNONYM_GROUPS (двунаправленно)."""
        q = question.lower()
        extra = []
        for triggers, expansions in self._SYNONYM_GROUPS:
            if any(trig in q for trig in triggers):
                for exp in expansions:
                    if exp not in extra and exp not in q:
                        extra.append(exp)
        return (question + " " + " ".join(extra)).strip()

    @staticmethod
    def _normalize_query(question: str) -> str:
        """Убирает слова-паразиты («скажи», «пожалуйста», «вопрос»...), не несущие
        юридического смысла, чтобы BM25/эмбеддинги не шумели."""
        return re.sub(r"\s+", " ", _FILLER_RE.sub(" ", question)).strip(" ,;").strip()

    def _rewrite_query(self, question: str) -> str:
        """LLM-переформулировка разговорного вопроса в юр. поисковый запрос (GigaChat).

        При сбое возвращает исходный вопрос — грациозная деградация до синонимов/оригинала.
        """
        prompt = (
            "Ты — помощник юридического поиска по российским кодексам (ТК, ГК, НК, СК, ЖК, "
            "УК и др.). Переформулируй вопрос обычного человека в формальные юридические "
            "термины (существительные в именительном падеже). Верни ТОЛЬКО поисковый запрос "
            "в одну строку из 3-7 ключевых слов, без пояснений и без кавычек.\n\nВопрос: " + question
        )
        try:
            out = self.llm.invoke(prompt)
            text = (out.content if hasattr(out, "content") else str(out)).strip()
            text = text.strip("\"' \n")  # отсекаем артефакты форматирования
            return text or question
        except Exception as e:
            log.warning("LLM query rewrite не удался, используем исходный запрос: %s", e)
            return question

    def _prepare_query(self, question: str) -> list:
        """Готовит варианты запроса: исходный (сырой) + нормализованный + LLM-переформулировка
        + синоним-расширения.

        ВАЖНО: первым вариантом (qi==0) идёт СЫРОЙ вопрос — именно по нему считается гейт
        релевантности (orig_best) в `_retrieve_hybrid_multi`. Нормализация (удаление
        «скажи/как правильно») иногда УХУДШАЕТ эмбеддинг: «как правильно уволиться с работы»
        после нормализации → «уволиться с работы» даёт дистанцию ~1.35 и отсекался, а сырой
        запрос — ~1.20 и релевантен. Гейт по сырому запросу честно отсекает нерелевантное
        («пенсия» ~1.30, «соседи шумят» ~1.53), не давая LLM-rewrite «спасать» шум.
        Кэш — по нормализованному тексту.
        """
        raw = question
        question = self._normalize_query(question)
        cached = self._query_cache.get(question)
        if cached is not None:
            return cached
        rewritten = self._rewrite_query(question)
        variants = [raw, question, rewritten, self._expand_query(question), self._expand_query(rewritten)]
        seen, uniq = set(), []
        for v in variants:
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                uniq.append(v)
        if len(self._query_cache) > 128:  # защита от разрастания кэша между запросами
            self._query_cache.clear()
        self._query_cache[question] = uniq
        return uniq

    def _keyword_search(self, question: str, k: int):
        q = self._vectorizer.transform([question])
        sims = (self._tfidf_matrix @ q.T).toarray().ravel()
        order = np.argsort(sims)[::-1]
        out = []
        for i in order:
            if sims[i] <= 0:
                break
            meta = self._chunk_metadatas[i] if i < len(self._chunk_metadatas) else {}
            out.append(Document(page_content=self._chunk_texts[i], metadata=meta or {}))
            if len(out) >= k:
                break
        return out

    # Порог косинусной дистанции Chroma: выше = не релевантно.
    # Откалибровано под Giga-Embeddings-instruct-480M-0826 (1024-dim, norm=True):
    # на замере 7 релевантных запросов best_dist = 0.88..1.25, пограничные юр-формулировки
    # (закон о тишине 1.20, нарушение покоя 1.33) и 4 мусорных = 1.37..1.65.
    # Порог иррелевантности. Калибровка по Giga-эмбеддингам:
    #   REL_max ≈ 1.2475 (самый дальний релевантный чанк)
    #   пограничный ≈ 1.330, NOISE_min ≈ 1.3751.
    # Берём 1.28 — чуть выше REL_max, но ниже пограничного: гарантированно пропускает
    # все релевантные ответы и ЖЁСТКО отсекает «шум». Например, запрос «пенсия» даёт
    # global_min ≈ 1.31 (в 26 кодексах нет ФЗ-400 о пенсиях) и должен уходить в
    # «нет ответа», а не провоцировать галлюцинации модели. Если начнёт отсекать
    # нужное — поднимите (напр. 1.30), если пролезет мусор — опустите (напр. 1.25).
    _MAX_IRRELEVANT_DISTANCE = 1.28
    # Порог СЕМАНТИЧЕСКОЙ близости исходного запроса для гейта в _retrieve_hybrid_multi.
    # Используется в комбинации с BM25 (OR): отвергаем только если ни ключей, ни
    # смысловой близости. 1.5 по CPU-замерам: пенсия≈1.30/неуплата≈1.1 проходят,
    # соседи≈1.6/борщ≈1.7 — нет. См. NOTES (продолжение 5, фикс гейта).
    _SEMANTIC_ACCEPT_DISTANCE = 1.5

    def _search_by_vector(self, vector, k: int):
        """Поиск по готовому вектору через внутренний API Chroma.

        У Chroma нет публичного similarity_search_with_score_by_vector, поэтому
        идём через _collection.query и восстанавливаем Document из метаданных.
        Возвращает список (Document, distance). При сбое — [].
        """
        try:
            v = [float(x) for x in vector]
            n = min(k, self.vectorstore._collection.count()) if k else k
            res = self.vectorstore._collection.query(
                query_embeddings=[v],
                n_results=max(1, n),
                include=["documents", "metadatas", "distances"],
            )
            out = []
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                out.append((Document(page_content=doc, metadata=meta or {}), float(dist)))
            return out
        except Exception as e:
            log.warning("Поиск по вектору не удался: %s", e)
            return []

    def _retrieve_hybrid_multi(self, queries: list, k: int, hyde_vector=None):
        """Гибридный поиск по НЕСКОЛЬКИМ вариантам запроса + RRF-мерж.

        Для каждого варианта (оригинал / LLM-переформулировка / синоним-расширения)
        считаем векторный и BM25-поиск, затем сливаем кандидатов через Reciprocal
        Rank Fusion (оригинальный запрос чуть перевешивает).

        Фильтр ложных источников (детерминированный, не зависит от формулировки LLM):
          1) ни один вариант не дал BM25-попаданий — в корпусе нет ключевых терминов;
          2) самый релевантный кандидат всё же слабо похож (векторная дистанция выше
             порога `_MAX_IRRELEVANT_DISTANCE`) — значит контекста нет, показывать нечего.
        В обоих случаях возвращаем пусто → LLM честно «нет ответа», без ложных цитат.
        """
        n = len(self._chunk_texts)
        top_n = min(20, n)
        if self._vectorizer is None:
            # Нет BM25-индекса — только векторный поиск по первому варианту
            return self.vectorstore.similarity_search(queries[0], k=top_n)[:k]

        fused = {}
        all_docs = {}
        best_dist = {}  # page_content -> лучшая (минимальная) векторная дистанция по вариантам
        orig_best = None  # лучшая векторная дистанция ИСХОДНОГО запроса (qi == 0)
        kw_hits = 0
        vs_w, kw_w = 1.0, 2.5  # BM25 чуть весомее — ловит морфологию в русском
        for qi, q in enumerate(queries):
            variant_boost = 1.0 + (0.3 if qi == 0 else 0.0)
            vs_res = self.vectorstore.similarity_search_with_score(q, k=top_n)
            kw_docs = self._keyword_search(q, k=top_n)
            kw_hits += len(kw_docs)
            if qi == 0:
                for _d, _dist in vs_res:
                    if orig_best is None or _dist < orig_best:
                        orig_best = _dist
            for rank, (d, dist) in enumerate(vs_res):
                c = d.page_content
                fused[c] = fused.get(c, 0.0) + vs_w * variant_boost / (rank + 1 + 60)
                if c not in best_dist or dist < best_dist[c]:
                    best_dist[c] = dist
            for rank, d in enumerate(kw_docs):
                c = d.page_content
                fused[c] = fused.get(c, 0.0) + kw_w * variant_boost / (rank + 1 + 60)
            for d, _ in vs_res:
                c = d.page_content
                # сохраняем метаданные: при совпадении текста отдаём версию с метаданными
                if c not in all_docs or not all_docs[c].metadata:
                    all_docs[c] = d
            for d in kw_docs:
                c = d.page_content
                if c not in all_docs or not all_docs[c].metadata:
                    all_docs[c] = d

        # HyDE (опц.): доп. векторный поиск по эмбеддингу гипотетического фрагмента
        if hyde_vector is not None:
            try:
                hv_res = self._search_by_vector(hyde_vector, top_n)
                for rank, (d, dist) in enumerate(hv_res):
                    c = d.page_content
                    fused[c] = fused.get(c, 0.0) + vs_w / (rank + 1 + 60)
                    if c not in best_dist or dist < best_dist[c]:
                        best_dist[c] = dist
                    if c not in all_docs or not all_docs[c].metadata:
                        all_docs[c] = d
            except Exception as e:
                log.warning("HyDE vector search не удался: %s", e)

        # Фильтр ложных источников (детерминированный, без хрупкого порога по дистанции):
        # Принимаем, если ВЫПОЛНЯЕТСЯ ХОТЯ БЫ ОДНО:
        #   1) был хотя бы один BM25-хит (ключевые термины есть в корпусе, пусть и в
        #      другой форме — морфология лечится синонимами/расширением запроса);
        #   2) ИСХОДНЫЙ запрос семантически близок корпусу (raw-дистанция ≤ порога).
        # Отвергаем ТОЛЬКО когда ОБА не сработали (напр. «как приготовить борщ»,
        # «соседи шумят» — в корпусе нет ни ключей, ни близких по смыслу фрагментов).
        # Порог 1.5 подобран по CPU-замерам (идентично ВМ): пенсия≈1.30 и неуплата≈1.1
        # ПРОХОДЯТ (дальше их галлюцинации ловит grounding), а соседи≈1.6/борщ≈1.7 —
        # НЕТ. Использовать raw-дистанцию (а не HyDE/rewrite), т.к. те искусственно
        # занижают расстояние даже для «тема упомянута, но ответа нет».
        if kw_hits == 0 and (orig_best is None or orig_best > self._SEMANTIC_ACCEPT_DISTANCE):
            return []

        ranked = sorted(fused.keys(), key=lambda t: fused[t], reverse=True)
        return [all_docs[t] for t in ranked[:k]]

    # --- Публичный API движка ---
    def ask(self, question: str, max_retries: int = 3) -> str:
        last_error = None
        for _ in range(max_retries + 1):
            try:
                answer = (self.rag_chain.invoke(question) or "").strip()
            except Exception as e:
                last_error = e
                answer = ""
            if answer:
                return answer
            last_error = ValueError(
                "LLM вернул пустой ответ. Вероятная причина — reasoning-модель без "
                "отключённого режима размышлений. Используйте не-reasoning модель "
                "(GigaChat-2 / GigaChat-Pro) или задайте LLM_EXTRA_BODY."
            )
        raise last_error

    @staticmethod
    def _is_no_answer(answer: str) -> bool:
        """True, если модель честно сообщила, что в документах нет ответа.

        GigaChat парафразирует каноническую фразу, поэтому ловим несколько
        формулировок («нет ответа» / «нет информации» / «не содержится» и т.п.),
        чтобы не показывать ложные источники рядом с «нет ответа».
        """
        a = (answer or "").strip().lower()
        if not a:
            return True
        if a.startswith("в предоставленн"):
            return True
        negatives = [
            "нет ответа", "нет прямого ответа", "нет информации", "нет данных",
            "нет сведений", "не содержится", "не нашлось", "не удалось найти",
            "отсутствует информация", "в контексте нет", "в предоставленных документах нет",
            "в доступных документах нет", "контекст не содержит", "не упоминается",
            "не приводится", "нет упоминаний",
        ]
        if any(ph in a for ph in negatives):
            return True
        # «нет» + одно из ключевых слов поблизости (информация/ответ/сведения/данные…).
        # НЕ добавляем сюда «закон/статья/кодекс» — иначе ложно санитизируем правильные
        # ответы вида «Нет, ст. 81 ТК не применяется…».
        if "нет" in a and any(k in a for k in ["информаци", "ответ", "сведени", "данн", "упоминан", "материал"]):
            return True
        return False

    def _is_grounded(self, answer: str, docs) -> bool:
        """Grounding-санитайзер: ответ не должен цитировать законы/кодексы вне retrieved docs.

        Корпус — 26 кодексов РФ (федеральные законы/ФЗ в нём отсутствуют). Поэтому:
          - любая ссылка на «ФЗ-...» в ответе = галлюцинация (ФЗ в корпусе нет);
          - упоминание кодекса/конституции, которого НЕТ среди найденных документов =
            галлюцинация (модель додумала чужой источник);
          - если ответ вообще не называет кодекс/ФЗ — считаем укоренённым (не пустой).
        Возвращает True, если ответ можно показывать, иначе False (→ «нет ответа»).
        """
        a = (answer or "").lower()
        # допустимые кодексы собираем из МЕТАДАННЫХ найденных документов
        allowed = set()
        for d in docs:
            src = (d.metadata.get("source") or "").lower()
            for _, file_kw in self._CODEX_ALIASES:
                if file_kw in src:
                    allowed.add(file_kw)
        # Корпус — 26 кодексов РФ; ФЕДЕРАЛЬНЫХ ЗАКОНОВ в нём нет. Любая ссылка на ФЗ/закон
        # с номером или датой в ответе — галлюцинация (модель додумала источник вне корпуса).
        if re.search(r"фз[\s\-]*\d+", a):                       # «ФЗ-400», «ФЗ 4468-I»
            return False
        if re.search(r"закон\w*\s+(российск\w*|федерац\w*|рф)", a):  # «Закон(ом/е) РФ / Российской Федерации…»
            return False
        if re.search(r"федеральн\w*\s+закон", a):               # «Федеральный закон…» (любой падеж)
            return False
        if re.search(r"закон\w*\s+от\s+\d", a):                 # «Закон от 12 февраля …»
            return False
        if re.search(r"\d+\s*-\s*фз", a):                       # «137-ФЗ»
            return False
        # какие кодексы упомянуты в самом ответе
        mentioned = set()
        for alias, file_kw in self._CODEX_ALIASES:
            if alias in a:
                mentioned.add(file_kw)
        # упомянут кодекс, которого нет среди найденных документов → галлюцинация
        for file_kw in mentioned:
            if file_kw not in allowed:
                return False
        return True

    def _ask_reinforced(self, question: str, docs) -> str:
        """Повторный запрос к LLM с усиленным промптом при ложно-отрицательном «нет ответа».

        Используется, когда retrieve() нашёл релевантный контекст, а модель всё равно
        ответила «нет ответа» (флуктуация GigaChat). Промпт жёстко требует ИЗВЛЕЧЬ ответ
        из Контекста и запрещает «нет ответа» при наличии хоть какой-то информации.
        Финальный ответ всё равно проходит grounding-проверку в ask_with_sources(), поэтому
        галлюцинации (вне-корпусные ФЗ/законы) отсекаются.
        """
        ctx = format_docs(docs)
        limit = self.settings.llm_max_context_chars
        if limit and len(ctx) > limit:
            ctx = ctx[:limit]
            if "\n" in ctx:
                ctx = ctx[: ctx.rfind("\n")]
            ctx = ctx + "\n…(контекст усечён по лимиту)"
        sys = (
            SYSTEM_INSTRUCTION
            + "\n\nУТОЧНЕНИЕ: отвечай ТОЛЬКО если в Контексте есть ПРЯМОЙ и КОНКРЕТНЫЙ ответ "
              "на ВОПРОС пользователя. Если Контекст лишь упоминает тему, но НЕ содержит "
              "ответа на сам вопрос (например, говорит о смежном аспекте) — отвечай ТОЛЬКО "
              "«нет ответа». Не додумывай, не обобщай и строго не выходи за пределы Контекста."
        )
        msg = [
            SystemMessage(content=sys),
            HumanMessage(content=f"Контекст:\n{ctx}\n\nВопрос: {question}"),
        ]
        try:
            out = self.llm.invoke(msg)
            return (out.content if hasattr(out, "content") else str(out)).strip()
        except Exception as e:
            log.warning("Усиленный повтор не удался: %s", e)
            return ""

    def ask_with_sources(self, question: str, max_retries: int = 3):
        """Возвращает (ответ, список источников) для цитирования в UI.

        Если модель не нашла ответ в контексте, источники не возвращаются —
        иначе UI показывал бы нерелевантные «ложные источники» (см. грабли в NOTES).
        """
        docs = self.retrieve(question)
        if not docs:
            # Детерминированная защита от галлюцинаций: если поиск не нашёл ни одного
            # релевантного фрагмента — сразу «нет ответа», НЕ обращаясь к LLM. Иначе
            # GigaChat-2 игнорирует инструкцию и додумывает ответ из памяти, цитируя
            # законы/ФЗ вне корпуса (кейс «пенсия»: ФЗ-400/ФЗ-350). Источники пусты.
            return CANONICAL_NO_ANSWER, []
        answer = self.ask(question, max_retries=max_retries)
        if self._is_no_answer(answer):
            # Ложно-отрицательный «нет ответа»: контекст найден, но GigaChat по флуктуации
            # не извлёк ответ. Один усиленный повтор с жёстким требованием отвечать СТРОГО
            # из Контекста и говорить «нет ответа», если прямого ответа нет. Если усиленный
            # ответ снова пустой ИЛИ содержит галлюцинацию (вне-корпусный ФЗ/закон) —
            # отдаём каноническое «нет ответа» без ложных источников.
            reinforced = self._ask_reinforced(question, docs)
            if self._is_no_answer(reinforced) or not self._is_grounded(reinforced, docs):
                return CANONICAL_NO_ANSWER, []
            answer = reinforced
        # Grounding-санитайзер: если модель процитировала кодекс/ФЗ вне найденных документов,
        # считаем ответ галлюцинацией и возвращаем «нет ответа» без ложных источников.
        if not self._is_grounded(answer, docs):
            log.info("Grounding-санитайзер: ответ не укоренён в retrieved docs (вопрос: %r)", question)
            return CANONICAL_NO_ANSWER, []
        sources = [
            {
                "source": d.metadata.get("source", ""),
                "page": d.metadata.get("page"),
                "snippet": d.page_content[:300],
            }
            for d in docs
        ]
        return answer, sources

    def _detect_article_query(self, question: str):
        """Если вопрос содержит «статья N»/«ст. N»/«N ст» → вернуть (номер, codex_hint|None)."""
        m = ART_QUERY_RE.search(question) or ART_QUERY_RE2.search(question)
        if not m:
            return None
        num = int(m.group(1))
        ql = question.lower()
        hint = None
        for kw, src in CODEX_HINTS:
            if kw in ql:
                hint = src
                break
        return num, hint

    def _retrieve_by_article(self, article_no: int, codex_hint):
        """Точный поиск по номеру статьи через metadata `article` (заполняется при индексации).

        Обходит нечёткий поиск и коллизии с номерами страниц; при указании кодекса фильтрует
        по имени файла. Чанки с заголовком самой статьи сортируются выше.
        """
        if not getattr(self, "_chunk_metadatas", None):
            return []
        out = []
        for i, meta in enumerate(self._chunk_metadatas):
            if meta.get("article") != article_no:
                continue
            if codex_hint and codex_hint not in (meta.get("source") or ""):
                continue
            out.append(Document(page_content=self._chunk_texts[i], metadata=meta or {}))
        # Приоритет — чанку с самим заголовком «СТАТЬЯ N.» (по всему тексту, не только первые 200 симв.)
        out.sort(key=lambda d: 0 if re.search(r"СТАТЬЯ\s*%d\." % article_no, d.page_content, re.IGNORECASE) else 1)
        return out

    def _hyde_vector(self, question: str):
        """HyDE: генерируем гипотетический фрагмент кодекса и берём его эмбеддинг как запрос.

        При сбое (нет LLM/сети) возвращает None → используется обычный поиск.
        """
        try:
            hypo = self.llm.invoke(
                "Напиши краткий гипотетический фрагмент из российского кодекса, который "
                "отвечает на вопрос. Только текст статьи, без пояснений.\n\nВопрос: " + question
            )
            text = (hypo.content if hasattr(hypo, "content") else str(hypo)).strip()
            if text:
                return self.embeddings.embed_query(text)
        except Exception as e:
            log.warning("HyDE не удался, используем обычный поиск: %s", e)
        return None

    def _rerank(self, query: str, docs: list, top_n: int) -> list:
        """Переранжирование кандидатов reranker-моделью (MiniLM) по близости к запросу.

        Берём больше кандидатов из гибридного поиска, затем упорядочиваем их по
        косинусной близости эмбеддингов reranker-модели к запросу. Это поднимает
        действительно релевантный чанк (напр. нужную статью кодекса) наверх контекста,
        что снижает шанс ложно-отрицательного ответа LLM. Если reranker недоступен —
        возвращаем исходных кандидатов (деградация без поломок).
        """
        if self._rerank_model is None or not docs:
            return docs[:top_n]
        try:
            q_emb = self._rerank_model.encode([query], normalize_embeddings=True)
            d_emb = self._rerank_model.encode(
                [d.page_content for d in docs], normalize_embeddings=True
            )
            sims = (d_emb @ q_emb[0]).tolist()
            order = sorted(range(len(docs)), key=lambda i: sims[i], reverse=True)
            return [docs[i] for i in order[:top_n]]
        except Exception as e:
            log.warning("Rerank не удался, возвращаем исходный порядок: %s", e)
            return docs[:top_n]

    def retrieve(self, question: str, k: int | None = None):
        k = k or self.settings.retriever_k
        qn = self._normalize_query(question)  # нормализуем для детекта статьи (убираем «скажи» и т.п.)
        # HyDE (опц., config.enable_hyde): гипотетический фрагмент → эмбеддинг запроса
        hyde_vec = self._hyde_vector(question) if getattr(self.settings, "enable_hyde", False) else None
        queries = self._prepare_query(question)
        # Берём расширенный пул кандидатов, затем переранжируем reranker-моделью до top_k
        cand_k = max(k * 3, 24)
        cand = self._retrieve_hybrid_multi(queries, cand_k, hyde_vector=hyde_vec)
        docs = self._rerank(question, cand, k)
        # Точный поиск по номеру статьи — ставим результаты выше нечёткого
        art = self._detect_article_query(qn)
        if art is not None:
            art_docs = self._retrieve_by_article(art[0], art[1])
            if art_docs:
                seen = {d.page_content for d in art_docs}
                merged = list(art_docs) + [d for d in docs if d.page_content not in seen]
                docs = merged[:k]
        return docs

    def contexts_for(self, question: str, k: int | None = None) -> list[str]:
        return [d.page_content for d in self.retrieve(question, k)]

    def list_documents(self) -> list[str]:
        if not os.path.exists(self.settings.docs_dir):
            return []
        return [f for f in os.listdir(self.settings.docs_dir)
                if f.lower().endswith((".txt", ".pdf", ".docx"))]

    def add_document(self, filename: str, content: bytes) -> dict:
        if not filename.lower().endswith((".txt", ".pdf", ".docx")):
            raise ValueError("Только .txt / .pdf / .docx файлы")
        os.makedirs(self.settings.docs_dir, exist_ok=True)
        path = os.path.join(self.settings.docs_dir, filename)
        with open(path, "wb") as f:
            f.write(content)
        return self.reindex()

    def remove_document(self, filename: str) -> dict:
        path = os.path.join(self.settings.docs_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {filename}")
        os.remove(path)
        return self.reindex()

    def reindex(self, docs_dir=None, chunk_size=1000, chunk_overlap=150) -> dict:
        """Полная переиндексация docs_dir -> коллекция Chroma (PDF/DOCX/TXT)."""
        with self._lock:
            docs_dir = docs_dir or self.settings.docs_dir
            if not os.path.exists(docs_dir):
                return {"error": f"Папка {docs_dir} не найдена"}
            documents = load_all_documents(docs_dir)
            if not documents:
                return {"error": "Нет документов для индексации (ожидаются .txt/.pdf/.docx)"}

            chunks = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ).split_documents(documents)

            # Вытаскиваем номер статьи из заголовка чанка в metadata (`article`),
            # чтобы работал точный поиск по номеру статьи (см. _retrieve_by_article).
            for _c in chunks:
                _m = ART_HEADER_RE.search(_c.page_content)
                if _m:
                    _c.metadata["article"] = int(_m.group(1))

            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass  # коллекции ещё нет — нормально при первом запуске

            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.client,
                collection_name=self.collection_name,
            )
            self._build()

        return {
            "status": "ok",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "message": f"Индексация завершена: {len(documents)} документов, {len(chunks)} чанков",
        }

    # --- Авто-переиндексация при изменении docs/ ---
    def _docs_signature(self) -> str:
        docs_dir = self.settings.docs_dir
        if not os.path.isdir(docs_dir):
            return "empty"
        newest, count = 0, 0
        for root, _dirs, files in os.walk(docs_dir):
            for f in files:
                if f.lower().endswith((".txt", ".pdf", ".docx")):
                    count += 1
                    try:
                        m = os.path.getmtime(os.path.join(root, f))
                    except OSError:
                        continue
                    if m > newest:
                        newest = m
        return f"{count}:{newest}"

    def _reindex_safe(self) -> dict:
        lock_path = os.path.join(self.settings.chroma_dir, ".reindex.lock")
        os.makedirs(self.settings.chroma_dir, exist_ok=True)
        from filelock import FileLock

        flock = FileLock(lock_path, timeout=300)
        with flock:
            current = self._docs_signature()
            sig_file = os.path.join(self.settings.chroma_dir, ".docs_sig")
            shared = ""
            if os.path.exists(sig_file):
                try:
                    shared = open(sig_file, encoding="utf-8").read().strip()
                except OSError:
                    shared = ""
            if current and current == shared:
                with self._lock:
                    self._build()
                return {"status": "reloaded", "message": "Индекс перезагружен без повторной переиндексации"}
            result = self.reindex()
            try:
                with open(sig_file, "w", encoding="utf-8") as fh:
                    fh.write(current or "empty")
            except OSError:
                pass
            return result

    def _wlog(self, msg: str) -> None:
        try:
            import datetime
            os.makedirs("logs", exist_ok=True)
            with open("logs/watcher.log", "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.datetime.now().isoformat()} {msg}\n")
        except OSError:
            pass

    def start_watcher(self) -> None:
        if getattr(self, "_watcher_running", False):
            return
        try:
            from watchfiles import watch
        except ImportError:
            log.warning("watchfiles не установлен — авто-reindex отключён")
            return

        docs_dir = self.settings.docs_dir
        os.makedirs(docs_dir, exist_ok=True)
        self._watcher_running = True
        log.disabled = False
        log.info("Авто-reindex: слежу за изменениями в %s", docs_dir)
        self._wlog(f"watcher started for {docs_dir} (pid={os.getpid()})")

        def _loop():
            import time
            try:
                for _changes in watch(
                    docs_dir,
                    step=500,
                    watch_filter=lambda change, path: str(path).lower().endswith((".txt", ".pdf", ".docx")),
                ):
                    time.sleep(2)
                    if not getattr(self, "_watcher_running", False):
                        break
                    log.info("Авто-reindex: изменение в %s — пересобираю индекс…", docs_dir)
                    self._wlog("change detected")
                    try:
                        res = self._reindex_safe()
                        log.info("Авто-reindex: %s", res.get("message"))
                        self._wlog(f"reindex done: {res.get('message')}")
                    except Exception:
                        log.exception("Авто-reindex: ошибка переиндексации")
                        self._wlog("reindex ERROR")
            except Exception:
                log.exception("Авто-reindex: watcher остановлен")
                self._wlog("watcher stopped with error")
                self._watcher_running = False

        threading.Thread(target=_loop, daemon=True).start()
