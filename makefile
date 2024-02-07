format:

check-format:


test:
	make unit-test
	make acceptance-test

unit-test:
	pytest -v --cov=sae_training/ --cov-report=term-missing --cov-branch tests/unit

acceptance-test:
	pytest -v --cov=sae_training/ --cov-report=term-missing --cov-branch tests/acceptance
