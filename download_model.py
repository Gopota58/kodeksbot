"""
ВНИМАНИЕ: автоматическое скачивание отключено.

Модель rubert-tiny2 уже находится локально в ./models/rubert-tiny2
(веса model.safetensors, pytorch_model.bin, tinybert-ru-labse-adapter-v2.pt).
Проект настроен на строго локальную загрузку (HF_HUB_OFFLINE=1 в config.py),
поэтому никакая авто-загрузка из сети не производится.

Этот скрипт оставлен только для справки: он скачает модель лишь в том случае,
если локальная папка отсутствует, и только при явном запуске
`python download_model.py`. Автоматически нигде не вызывается.
"""
from pathlib import Path

from config import settings

if __name__ == "__main__":
    local = Path(settings.model_dir)
    if (local / "config.json").exists():
        print(f"Модель уже присутствует локально в {local}. Скачивание не требуется.")
    else:
        # Локальной модели нет — скачиваем (требует доступа к Hugging Face).
        from huggingface_hub import snapshot_download

        print(
            f"Локальная модель не найдена. Загрузка {settings.embedding_model_id} -> {settings.model_dir}"
        )
        path = snapshot_download(
            repo_id=settings.embedding_model_id,
            local_dir=settings.model_dir,
        )
        print(f"Готово: {path}")
