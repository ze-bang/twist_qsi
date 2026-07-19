.PHONY: validate nonperturbative nonperturbative-m3 nonperturbative-m4 figures test paper note all clean

PYTHON ?= python

validate:
	PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) campaign/run_validation.py

nonperturbative:
	PYTHONPATH=src $(PYTHON) campaign/run_nonperturbative.py

nonperturbative-m3:
	PYTHONPATH=src $(PYTHON) campaign/run_nonperturbative.py --max-grid 3

nonperturbative-m4:
	PYTHONPATH=src $(PYTHON) campaign/run_nonperturbative.py --max-grid 4

figures: nonperturbative-m4
	PYTHONPATH=src $(PYTHON) campaign/make_figures.py

test:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest

paper: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/supplement.tex

note: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd docs/projected_band_protocol_note.tex

all: validate test paper note

clean:
	latexmk -C -cd paper/main.tex
	latexmk -C -cd paper/supplement.tex
	latexmk -C -cd docs/projected_band_protocol_note.tex
