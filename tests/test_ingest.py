"""
Тест переиндексации: убеждаемся, что движок строит непустую коллекцию.
Не требует LLM (нужны только локальные эмбеддинги).
"""
import pytest
from rag.engine import RAGEngine


def test_reindex_builds_nonempty_collection():
    engine = RAGEngine()
    result = engine.reindex()
    assert result["status"] == "ok"
    assert result["documents_loaded"] > 0
    assert result["chunks_created"] > 0
    # после реиндекса ретривер должен что-то находить
    docs = engine.retriever.invoke("чем кормить кота")
    assert len(docs) > 0
