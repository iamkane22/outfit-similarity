.PHONY: install data embed evaluate serve app test lint typecheck check

install:
	uv sync --extra dev --extra app --extra report

data:
	uv run outfit-search ingest

embed:
	uv run outfit-search embed --encoder clip
	uv run outfit-search embed --encoder fashion-clip

evaluate:
	uv run outfit-search evaluate --encoder clip
	uv run outfit-search evaluate --encoder fashion-clip

serve:
	uv run outfit-search serve --encoder fashion-clip

app:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest --cov=outfit_search --cov-report=term-missing

lint:
	uv run ruff check src tests app
	uv run ruff format --check src tests app

typecheck:
	uv run mypy

check: lint typecheck test
