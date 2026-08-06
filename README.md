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

Native text is used by default. Selective Tesseract OCR can recover text from
scanned pages while preserving page-level extraction provenance:

```python
from sightcite.ingestion import TesseractOcrBackend
from sightcite.pipelines import TextRetrievalPipeline
from sightcite.retrieval import BgeTextEmbedder

pipeline = TextRetrievalPipeline(
    "paper.pdf",
    BgeTextEmbedder(),
    ocr_backend=TesseractOcrBackend(),
)
```

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

## Reproducible baseline experiment

The synthetic baseline contains two native-text pages and one image-only page.
It demonstrates the retrieval difference between native extraction and selective
OCR without requiring an external paper dataset.

Generate the benchmark PDF:

```bash
python scripts/generate_baseline_fixture.py
```

Run both retrieval configurations:

```bash
python scripts/run_baseline_comparison.py --device cpu
```

The experiment writes detailed reports under `examples/baseline/reports/`.
With the default BGE model, the expected aggregate result is:

| System | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| `bge-text` | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| `bge-text-ocr` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The native system cannot retrieve the image-only microscopy page. Selective
OCR recovers that page while preserving retrieval performance on the
native-text pages.

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

Run the opt-in real-Tesseract smoke test:

```bash
SIGHTCITE_RUN_OCR_TESTS=1 pytest -m ocr -v --no-cov
```
