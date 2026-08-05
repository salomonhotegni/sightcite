# SightCite

SightCite is an evidence-grounded visual retrieval-augmented generation system
for scientific papers.

It retrieves relevant PDF pages using textual and visual representations, answers
questions with a vision-language model, and returns page-level evidence.

## Status

SightCite is under active development.

## Native-text retrieval

Build a page-aware text index for a PDF:

```python
from sightcite.pipelines import TextRetrievalPipeline
from sightcite.retrieval import BgeTextEmbedder

embedder = BgeTextEmbedder()
pipeline = TextRetrievalPipeline("paper.pdf", embedder)

for result in pipeline.search("What is the main contribution?", top_k=3):
    print(result.rank, result.chunk.page_number, result.score)
    print(result.chunk.text)
```

The current text baseline requires PDFs containing embedded text. OCR support for
scanned pages is planned.

## Retrieval benchmark CLI

Run a versioned page-retrieval benchmark:

```bash
sightcite benchmark benchmark.json \
  --output reports/text-retrieval.json \
  --device cpu
```

The command prints Recall@1, Recall@3, Recall@5, and mean reciprocal rank,
then writes aggregate and per-query results to JSON.

See all options:

```bash
sightcite benchmark --help
```

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

Run the opt-in real-model smoke test:

```bash
SIGHTCITE_RUN_MODEL_TESTS=1 pytest -m model -v --no-cov
```