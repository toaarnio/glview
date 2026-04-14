deps:
	uv sync --active --extra dev

lint:
	uv run --active ruff check glview/[^a]*.py

install: lint
	rm -rf dist || true
	uv build
	uv pip install --upgrade dist/*.whl
	@glview --version

release:
	@echo "Publishing is handled by GitHub Actions with PyPI Trusted Publishing."
	@echo "Create a GitHub Release to trigger the publish workflow."

.PHONY: deps lint install release
