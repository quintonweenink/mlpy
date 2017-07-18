init:
	pip install -r requirements.txt

run:
	python src/main.py

test:
	py.test tests

.PHONY: init test
