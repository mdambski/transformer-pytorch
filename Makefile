.PHONY: lint

lint:
	ruff check --fix-only
	ruff format .

test:
	pytest tests
