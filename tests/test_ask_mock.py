"""
Сквозной smoke-тест генерации БЕЗ внешнего LLM.

Подменяем чат-модель фиктивной, чтобы проверить весь пайплайн ask()
(retrieve -> prompt -> LLM -> parse) независимо от доступности LLM-эндпоинта.
Это доказывает корректность цепочки и позволяет гонять тест в CI без GPU/сервера.
"""
import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

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
    # гарантируем, что индекс доступен (если соседние тесты его перестроили)
    try:
        engine.retriever.invoke("тест индекса")
    except Exception:
        engine.reindex()

    answer = engine.ask("Сколько раз в день кормить кота?")
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "КОТ-МАРКЕР" in answer
