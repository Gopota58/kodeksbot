"""
CLI для построения индекса: `python ingest.py`.

Логика переиндексации инкапсулирована в RAGEngine (rag/engine.py).
Читает PDF/DOCX/TXT из data/docs/ и строит коллекцию Chroma.
"""
from rag.engine import RAGEngine

if __name__ == "__main__":
    engine = RAGEngine()
    result = engine.reindex()
    print(result.get("message", result))
