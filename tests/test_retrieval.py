"""
Тест качества retrieval: на типичный вопрос должны возвращаться релевантные
фрагменты (проверяем по ключевым словам из базы знаний).

Движок создаётся лениво (через fixture), а не на уровне модуля: иначе он
захватывает коллекцию chroma на момент импорта, и соседний test_ingest,
который вызывает reindex() и пересоздаёт коллекцию, делает этот движок
несвязанным (устаревшим) — отсюда intermittent-ошибки chroma.
"""
import pytest

from rag.engine import RAGEngine


@pytest.fixture(scope="module")
def engine():
    e = RAGEngine()
    # Гарантируем, что индекс актуален: если коллекция отсутствует
    # (соседний тест вызвал reindex и удалил старую, либо это чистый клон),
    # перестраиваем один раз.
    try:
        e.retriever.invoke("проверка индекса")
    except Exception:
        e.reindex()
    return e


def test_retriever_returns_relevant_chunks(engine):
    query = "чем кормить рыжего кота и сколько раз в день"
    docs = engine.retriever.invoke(query)
    assert len(docs) > 0
    text = " ".join(d.page_content.lower() for d in docs)
    # ожидаем, что среди топ-чанков будет упоминание кормления
    assert ("корм" in text) or ("кота" in text) or ("питан" in text)


def test_retriever_context_helper(engine):
    contexts = engine.contexts_for("где должен быть лоток для кота")
    assert isinstance(contexts, list)
    assert len(contexts) > 0
    assert all(isinstance(c, str) for c in contexts)
