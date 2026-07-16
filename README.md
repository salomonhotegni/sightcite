# SightCite

SightCite is an evidence-grounded visual retrieval-augmented generation system
for scientific papers.

It retrieves relevant PDF pages using textual and visual representations, answers
questions with a vision-language model, and returns page-level evidence.

## Status

SightCite is under active development.

## Development

SightCite requires Python 3.11.

Install the package and development tools:

```bash
python -m pip install --editable ".[dev]"
```

Run the development checks:

```bash
ruff check .
ruff format --check .
mypy
pytest
```
