init:
	pip install -r requirements.txt

run-nn:
	python src/neuralNetwork/main.py

run-pso:
	python src/main.py

test:
	py.test tests

.PHONY: init test
