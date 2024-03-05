PY_SOURCE=qtext test.py

dev:
	@pip install -e .[dev]

lint:
	@ruff check ${PY_SOURCE}

format:
	@ruff check --fix ${PY_SOURCE}
	@ruff format ${PY_SOURCE}

clean:
	@-rm -rf dist build */__pycache__ *.egg-info

build:
	@python -m build

test:
	@pytest -v tests
