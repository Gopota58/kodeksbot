"""
RAGAS-стиль оценка качества RAG-движка на золотом наборе (golden_dataset.json).

Запуск из корня проекта:
    python -m evaluation.run_eval
    # или:  python evaluation/run_eval.py

Использует тот же RAGEngine и тот же локальный LLM/эмбеддинги, что и продакшен,
поэтому оценка полностью оффлайн и бесплатна. Результаты печатаются в консоль
и сохраняются в evaluation/results.json.
"""
import json
import os
import sys

# гарантируем, что корень проекта в sys.path (для запуска как скрипта)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# вывод можем содержать эмодзи/кириллицу — явно UTF-8 (иначе падает на cp1251 в Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from rag.engine import RAGEngine
from evaluation.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall_lexical,
)


def main():
    engine = RAGEngine()
    llm = engine.llm

    data_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(data_path, encoding="utf-8") as f:
        dataset = json.load(f)

    rows = []
    for item in dataset:
        question = item["question"]
        reference = item["reference_answer"]

        # прогон через реальный пайплайн движка
        answer = engine.ask(question)
        contexts = engine.contexts_for(question)

        f_score = faithfulness(llm, contexts, answer)
        ar_score = answer_relevancy(llm, question, answer)
        cp_score = context_precision(llm, question, contexts)
        cr_score = context_recall_lexical(reference, contexts)

        rows.append({
            "question": question,
            "answer": answer,
            "faithfulness": f_score,
            "answer_relevancy": ar_score,
            "context_precision": cp_score,
            "context_recall_lexical": round(cr_score, 4),
        })
        print(f"• {question}")
        print(f"  ответ: {answer[:120]}...")
        print(f"  faithfulness={f_score} answer_relevancy={ar_score} "
              f"context_precision={cp_score} context_recall_lexical={cr_score:.2f}")

    # агрегация (исключаем None из LLM-судей)
    def _mean(values):
        vals = [v for v in values if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    summary = {
        "n_samples": len(rows),
        "faithfulness": _mean([r["faithfulness"] for r in rows]),
        "answer_relevancy": _mean([r["answer_relevancy"] for r in rows]),
        "context_precision": _mean([r["context_precision"] for r in rows]),
        "context_recall_lexical": _mean([r["context_recall_lexical"] for r in rows]),
    }

    print("\n=== ИТОГОВЫЕ МЕТРИКИ ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": rows}, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено в {out_path}")


if __name__ == "__main__":
    main()
