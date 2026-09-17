"""
Сквозной smoke-тест генерации БЕЗ внешнего LLM.

Подменяем чат-модель фиктивной, чтобы проверить весь пайплайн ask()
(retrieve -> prompt -> LLM -> parse) независимо от доступности LLM-эндпоинта.
Это доказывает корректность цепочки и позволяет гонять тест в CI без GPU/сервера.
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from rag.engine import RAGEngine


class MarkerChatModel(BaseChatModel):
    """Минимальная фейковая чат-модель: возвращает маркер, подтверждая,
    что цепочка дошла до LLM и корректно распарсила ответ."""
    marker: str = "КОТ-МАРКЕР"

    @property
    def _llm_type(self) -> str:
        return "marker"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=f"Ответ на основе контекста: {self.marker}"))]
        )


def test_ask_pipeline_without_real_llm():
    engine = RAGEngine(llm=MarkerChatModel())
    # Гарантируем непустой индекс. Проверяем именно наполнение коллекции: на пустой
    # коллекции retriever.invoke() не падает (Chroma с k>0 просто отдаёт []), поэтому
    # прежняя проверка «через invoke + except reindex» ничего не гарантировала —
    # и тест падал на пустом индексе в CI, где chroma_db создаётся с нуля.
    if engine.vectorstore._collection.count() == 0:
        engine.reindex()
    assert engine.vectorstore._collection.count() > 0

    answer = engine.ask("Сколько раз в день кормить кота?")
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "КОТ-МАРКЕР" in answer


def test_ask_on_empty_index_returns_no_answer():
    """Регрессия: пустой индекс не должен валить движок исключением.

    До фикса Chroma падала на `n_results=0` (TypeError) внутри гибридного поиска,
    а ask() маскировал это невнятным «LLM вернул пустой ответ».
    """
    engine = RAGEngine(llm=MarkerChatModel())
    engine._chunk_texts = []
    engine._chunk_metadatas = []
    engine._vectorizer = None

    assert engine.retrieve("Сколько раз в день кормить кота?") == []
    answer, sources = engine.ask_with_sources("Сколько раз в день кормить кота?")
    assert sources == []
    assert "нет ответа" in answer.lower()
