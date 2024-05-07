# Contributing

Contributions are welcome! To get setup for development, follow the instructions below.

## Setup

Make sure you have [poetry](https://python-poetry.org/) installed, clone the repository, and install dependencies with:

```bash
poetry install
```

## Testing, Linting, and Formatting

This project uses [pytest](https://docs.pytest.org/en/stable/) for testing, [flake8](https://flake8.pycqa.org/en/latest/) for linting, [pyright](https://github.com/microsoft/pyright) for type-checking, and [black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/) for formatting.

If you add new code, it would be greatly appreciated if you could add tests in the `tests/unit` directory. You can run the tests with:

```bash
make unit-test
```

Before commiting, make sure you format the code with:

```bash
make format
```

Finally, run all CI checks locally with:

```bash
make check-ci
```

If these pass, you're good to go! Open a pull request with your changes.

## Documentation

This project uses [mkdocs](https://www.mkdocs.org/) for documentation. You can see the docs locally with:

```bash
make docs-serve
```
If you make changes to code which requires updating documentation, it would be greatly appreciated if you could update the docs as well.
