.PHONY: dev api-dev package package-unpacked test test-cov test-full

DEV_DB ?= data/overmind.db


dev:
	npm install
	OVERMIND_DB=$(DEV_DB) npm run dev


api-dev:
	OVERMIND_DB=$(DEV_DB) uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


package:
	.venv/bin/pip install -e .[dev]
	npm install
	npm run dist


package-unpacked:
	.venv/bin/pip install -e .[dev]
	npm install
	npm run pack


test:
	pytest

test-cov:
	pytest --cov=app --cov-report=term-missing

test-full: test
