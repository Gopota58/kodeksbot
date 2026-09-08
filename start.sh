#!/usr/bin/env bash
# Кроссплатформенный запуск «Котобазы» (Linux / macOS)
set -e

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 не найден! Установите Python 3.10+."
  exit 1
fi

if [ ! -d venv ]; then
  echo "Создание виртуального окружения..."
  python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Подготовка модели эмбеддингов (при необходимости)..."
python download_model.py || echo "[info] Модель уже на месте или будет загружена автоматически."

echo "Запуск API-сервера на http://localhost:8000 ..."
uvicorn app:app --host 0.0.0.0 --port 8000
