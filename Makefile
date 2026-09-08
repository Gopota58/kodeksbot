.PHONY: install index run eval test docker-up clean

install:
	python -m venv venv && . venv/bin/activate && pip install -r requirements.txt

index:
	python ingest.py

run:
	uvicorn app:app --reload --port 8000

eval:
	python -m evaluation.run_eval

test:
	python -m pytest -q

docker-up:
	docker compose up --build

clean:
	rm -rf chroma_db __pycache__ */__pycache__
