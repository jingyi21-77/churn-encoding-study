.PHONY: install data test quick reproduce clean

install:
	pip install -r requirements.txt pytest

data:
	python -c "import sys; sys.path.insert(0,'src'); from churn_encoding.data import download; print(download())"

test:
	pytest -q

quick:            ## ~20 s smoke run (1 seed, tiny grid)
	python scripts/run_experiment.py --quick --output-dir results_quick

reproduce:        ## full study used for the README numbers
	python scripts/run_experiment.py --config configs/default.yaml

clean:
	rm -rf results_quick .pytest_cache **/__pycache__
