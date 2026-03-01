.PHONY: dev api-dev package package-linux package-mac package-win package-unpacked test test-cov test-full

DEV_DB ?= data/overmind.db

ifeq ($(OS),Windows_NT)
VENV_PYTHON ?= .venv/Scripts/python.exe
else
VENV_PYTHON ?= .venv/bin/python
endif


dev:
	npm install
	OVERMIND_DB=$(DEV_DB) npm run dev


api-dev:
	OVERMIND_DB=$(DEV_DB) uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


package:
	$(MAKE) package-linux


package-linux:
	$(VENV_PYTHON) -m pip install -e .[dev]
	npm install
	npm run dist:linux


package-mac:
	$(VENV_PYTHON) -m pip install -e .[dev]
	npm install
	npm run dist:mac


package-win:
	$(VENV_PYTHON) -m pip install -e .[dev]
	npm install
	npm run dist:win


package-unpacked:
	$(VENV_PYTHON) -m pip install -e .[dev]
	npm install
	npm run pack


test:
	pytest

test-cov:
	pytest --cov=app --cov-report=term-missing

test-full: test
