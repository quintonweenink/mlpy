init:
	pip install -r requirements.txt

run:
	python src/main.py

nn:
        python src/neuralNetwork/Test.py

test:
	py.test tests

.PHONY: init test
