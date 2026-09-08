#!/usr/bin/env bash
set -e

echo "[entrypoint] Подготовка модели эмбеддингов (при необходимости)..."
python download_model.py || echo "[entrypoint][warn] авто-загрузка модели не выполнена (ожидается локальная модель или HF-auto-download)"

echo "[entrypoint] Запуск API на :8000"
exec uvicorn app:app --host 0.0.0.0 --port 8000
