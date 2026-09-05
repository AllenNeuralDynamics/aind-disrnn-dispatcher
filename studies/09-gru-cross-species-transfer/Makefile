PYTHON ?= python
CACHE_ROOT ?= data-cache
PYTHONPATH := ../../code

.PHONY: all suite rebuild r1 r2

all: r1 r2

suite:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m external_bandit_datasets.cli all --cache-root $(CACHE_ROOT)

rebuild:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m external_bandit_datasets.cli all --cache-root $(CACHE_ROOT) --no-download

r1:
	$(PYTHON) analysis/report_matched_half.py

r2:
	$(PYTHON) analysis/report_author_baselines.py
