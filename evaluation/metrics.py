"""
RAGAS-стиль метрики качества RAG, реализованные in-house.

Реализованы те же концепции, что в библиотеке ragas, но без внешних
хрупких зависимостей и полностью оффлайн — поверх локального LLM и эмбеддингов
проекта:

  * faithfulness      — поддержан ли ответ только фактами из контекста (LLM-судья)
  * answer_relevancy  — отвечает ли ответ на вопрос (LLM-судья)
  * context_precision — релевантны ли найденные фрагменты вопросу (LLM-судья)
  * context_recall    — покрывает ли контекст эталонный ответ (лексический recall)

LLM-судья — это тот же локальный чат-модель; просим отвечать «да/нет» и
парсим. При неуверенном ответе возвращаем None (исключается из усреднения).
"""
import re

from langchain_core.messages import HumanMessage


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _judge(llm, prompt: str):
    """Возвращает 1.0 / 0.0 по ответу LLM «да/нет», либо None при неоднозначности."""
    try:
        out = _normalize(llm.invoke([HumanMessage(content=prompt)]).content)
    except Exception:
        return None
    if out.startswith("да") or out.startswith("yes") or out == "1":
        return 1.0
    if out.startswith("нет") or out.startswith("no") or out == "0":
        return 0.0
    if "да" in out or "yes" in out:
        return 1.0
    if "нет" in out or "no" in out:
        return 0.0
    return None


def faithfulness(llm, contexts, answer) -> float | None:
    ctx = "\n---\n".join(contexts)
    prompt = (
        f"Контекст:\n{ctx}\n\nУтверждение:\n{answer}\n\n"
        "Поддержано ли это утверждение ТОЛЬКО фактами из контекста? "
        "Ответь одним словом: да или нет."
    )
    return _judge(llm, prompt)


def answer_relevancy(llm, question, answer) -> float | None:
    prompt = (
        f"Вопрос: {question}\nОтвет: {answer}\n\n"
        "Отвечает ли этот ответ на поставленный вопрос? "
        "Ответь одним словом: да или нет."
    )
    return _judge(llm, prompt)


def context_precision(llm, question, contexts) -> float | None:
    ctx = "\n---\n".join(contexts)
    prompt = (
        f"Вопрос: {question}\n\nНайденные фрагменты:\n{ctx}\n\n"
        "Содержат ли эти фрагменты информацию, нужную чтобы ответить на вопрос? "
        "Ответь одним словом: да или нет."
    )
    return _judge(llm, prompt)


_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")


def _tokenize(text: str):
    return set(_TOKEN_RE.findall((text or "").lower()))


def context_recall_lexical(reference: str, contexts) -> float:
    """Лексический recall: какая доля содержательных слов эталонного ответа
    покрывается найденным контекстом (|ref ∩ ctx| / |ref|). Это именно recall,
    а не F1: контекст заведомо больше и лексически богаче короткого эталона,
    поэтому F1 штрафовал бы за размер и давал пессимистичные ~0.1."""
    ref_tok = _tokenize(reference)
    ctx_tok = _tokenize(" ".join(contexts))
    if not ref_tok:
        return 0.0
    return len(ref_tok & ctx_tok) / len(ref_tok)
