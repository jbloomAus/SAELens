format:
	poetry run black .
	poetry run isort .

check-format:
	poetry run flake8 .
	poetry run black --check .
	poetry run isort --check-only --diff .


test:
	make unit-test
	make acceptance-test

unit-test:
	poetry run pytest -v --cov=sae_training/ --cov-report=term-missing --cov-branch tests/unit

acceptance-test:
	poetry run pytest -v --cov=sae_training/ --cov-report=term-missing --cov-branch tests/acceptance
