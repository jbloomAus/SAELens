format:
	poetry run black .
	poetry run isort .

check-format:
	poetry run flake8 .
	poetry run black --check .
	poetry run isort --check-only --diff .

check-type:
	poetry run pyright .

test:
	make unit-test
	make acceptance-test

unit-test:
	poetry run pytest -v --cov=sae_lens/training/ --cov-report=term-missing --cov-branch tests/unit

acceptance-test:
	poetry run pytest -v --cov=sae_lens/training/ --cov-report=term-missing --cov-branch tests/acceptance

check-ci:
	make check-format
	make check-type
	make unit-test

docs-serve:
	poetry run mkdocs serve
