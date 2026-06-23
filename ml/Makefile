PY := python

.PHONY: setup download baseline train evaluate export smoke spike clean

setup:
	$(PY) -m pip install -r requirements.txt

download:
	$(PY) src/download_isic.py

baseline:
	$(PY) src/baseline.py

train:
	$(PY) src/train.py

evaluate:
	$(PY) src/evaluate.py

export:
	$(PY) src/export_onnx.py

# fast synthetic end-to-end check (no data download, no GPU)
smoke:
	$(PY) src/_smoke.py

# the full Phase 0 spike
spike: download baseline train evaluate export

clean:
	rm -rf data/ models/ reports/
