# On Windows without `make`, run the `python -m hcai ...` commands directly (see README).
PY ?= python

.PHONY: setup data docs train stability index eval test lint app cli monitor all
setup:
	$(PY) -m pip install -e ".[dev]"
data:
	$(PY) -m hcai data
docs:
	$(PY) scripts/fetch_docs.py
train:
	$(PY) -m hcai train
stability:
	$(PY) -m hcai stability
index:
	$(PY) -m hcai index
eval:
	$(PY) -m hcai eval
test:
	$(PY) -m pytest
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check . && $(PY) -m mypy
app:
	$(PY) -m streamlit run src/hcai/app/streamlit_app.py
cli:
	$(PY) -m hcai assess --help
monitor:
	$(PY) -m hcai monitor
all: setup lint test data docs train index eval monitor
