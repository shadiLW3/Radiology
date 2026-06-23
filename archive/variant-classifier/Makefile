PY := python

.PHONY: all setup download annotate features train evaluate pipeline clean

setup:
	$(PY) -m pip install -r requirements.txt

download:
	$(PY) src/download_clinvar.py

annotate:
	$(PY) src/annotate.py

features:
	$(PY) src/features.py

train:
	$(PY) src/train.py

evaluate:
	$(PY) src/evaluate.py

# full pipeline end-to-end
pipeline: download annotate features train evaluate

all: setup pipeline

clean:
	rm -rf data/*.csv data/*.gz models/* reports/*
