format:
	poetry run ruff format .
	poetry run ruff check --fix-only .

check-format:
	poetry run ruff check .
	poetry run ruff format --check .

check-type:
	poetry run pyright .

test:
	make unit-test
	make acceptance-test

unit-test:
	poetry run pytest -v --cov=sae_lens/ --cov-report=term-missing --cov-branch tests/unit

acceptance-test:
	poetry run pytest -v --cov=sae_lens/ --cov-report=term-missing --cov-branch tests/acceptance

check-ci:
	make check-format
	make check-type
	make unit-test

docstring-coverage:
	poetry run docstr-coverage sae_lens --skip-file-doc

docs-serve:
	poetry run mkdocs serve