import os
import tkinter as tk
from tkinter import scrolledtext, messagebox
import requests
import json
import threading

# --- КОНФИГУРАЦИЯ (можно переопределить через переменные окружения) ---
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/ask")
API_KEY = os.environ.get("API_KEY", "88888888")

# trust_env=False — ходить на localhost напрямую, минуя системный прокси
# (KiberportalX/др.), иначе запрос к локальному серверу уходит в прокси и
# возвращает 500. Сервер и бот уже так делают — клиент тоже.
_session = requests.Session()
_session.trust_env = False

class RAGClient:
    def __init__(self, root):
        self.root = root
        self.root.title("Котобаза — RAG-бот")
        self.root.geometry("600x500")
        self.root.resizable(True, True)

        # Поле вывода
        self.output = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12))
        self.output.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.output.config(state=tk.DISABLED)

        # Фрейм для ввода
        input_frame = tk.Frame(root)
        input_frame.pack(padx=10, pady=5, fill=tk.X)

        self.entry = tk.Entry(input_frame, font=("Arial", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.entry.bind("<Return>", lambda e: self.ask())

        self.send_btn = tk.Button(input_frame, text="Спросить", command=self.ask, width=10)
        self.send_btn.pack(side=tk.RIGHT)

        # Статус
        self.status_label = tk.Label(root, text="Готов", anchor="w", fg="gray")
        self.status_label.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Приветствие
        self.add_message("Бот", "Привет! Я Котобаза 😺\nЗадай мне вопрос по документам.")

    def add_message(self, sender, text):
        self.output.config(state=tk.NORMAL)
        self.output.insert(tk.END, f"{sender}: {text}\n\n")
        self.output.see(tk.END)
        self.output.config(state=tk.DISABLED)

    def ask(self):
        question = self.entry.get().strip()
        if not question:
            return
        self.entry.delete(0, tk.END)

        self.add_message("Вы", question)
        self.status_label.config(text="Думаю... 🐱")
        self.send_btn.config(state=tk.DISABLED)

        # Запрос в отдельном потоке, чтобы не блокировать GUI
        threading.Thread(target=self._request, args=(question,), daemon=True).start()

    def _request(self, question):
        try:
            headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
            payload = {"question": question}
            resp = _session.post(API_URL, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                answer = resp.json().get("answer", "Нет ответа")
            else:
                answer = f"Ошибка API: {resp.status_code}"
        except Exception as e:
            answer = f"Ошибка соединения: {e}"

        # Обновляем GUI в главном потоке
        self.root.after(0, lambda: self._on_response(answer))

    def _on_response(self, answer):
        self.add_message("Бот", answer)
        self.status_label.config(text="Готов")
        self.send_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGClient(root)
    root.mainloop()