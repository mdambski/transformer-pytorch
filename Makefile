.PHONY: lint
lint:
	ruff check --fix-only
	ruff format .

.PHONY: mypy
mypy:
	mypy .

.PHONY: test
test:
	pytest tests

.PHONY: install-deps
install-deps:
	poetry install --with dev
