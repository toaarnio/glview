deps:
	uv sync --active --extra dev

lint:
	uv run --active ruff check glview --exclude glview/argv.py

test: deps lint
	uv run --active pytest -vs

install: lint
	rm -rf dist || true
	uv build
	uv pip install --upgrade dist/*.whl
	uv run --active glview --version

macos-app:
	uv run --active python -m glview.glview --create-macos-app

windows-assoc-check:
	bash scripts/windows_assoc_check.sh

release:
	@echo "Publishing is handled by GitHub Actions with PyPI Trusted Publishing."
	@echo "Create a GitHub Release to trigger the publish workflow."

.PHONY: deps lint test install macos-app windows-assoc-check release
