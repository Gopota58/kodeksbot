"""
Явное скачивание модели эмбеддингов в `models/`.

По умолчанию проект работает СТРОГО офлайн: `config.py` выставляет
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, поэтому ни при старте сервера,
ни при индексации никаких обращений к Hugging Face не происходит.

Веса моделей в репозиторий не входят (`models/` в `.gitignore`) — их нужно
получить один раз:

    python download_model.py

Скрипт ничего не делает, если модель уже лежит на диске. Автоматически
(например, из `docker-entrypoint.sh`) он запускается только как страховка и
при отсутствии сети просто предупреждает, а не падает.
"""
import os
from pathlib import Path

from config import settings

if __name__ == "__main__":
    local = Path(settings.model_dir)
    if (local / "config.json").exists():
        print(f"Модель уже присутствует локально в {local}. Скачивание не требуется.")
    else:
        # Импорт config выставил HF_HUB_OFFLINE=1 — для явного скачивания
        # снимаем offline-флаги, иначе huggingface_hub откажется работать.
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        from huggingface_hub import snapshot_download

        print(
            f"Локальная модель не найдена. Загрузка {settings.embedding_model_id} -> {settings.model_dir}"
        )
        path = snapshot_download(
            repo_id=settings.embedding_model_id,
            local_dir=settings.model_dir,
        )
        print(f"Готово: {path}")
