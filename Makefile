.PHONY: dev test test-cov test-docker test-full

DEV_DB ?= data/overmind.db


dev:
	OVERMIND_DB=$(DEV_DB) uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


test:
	pytest

test-cov:
	pytest --cov=app --cov-report=term-missing


test-full: test test-docker


test-docker:
	OVERMIND_RUN_DOCKER_TESTS=1 pytest -k docker
