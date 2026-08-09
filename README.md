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
It compares native text, selective OCR, CLIP visual retrieval, and reciprocal
rank fusion without requiring an external paper dataset.

Generate the benchmark PDF:

```bash
python scripts/generate_baseline_fixture.py
```

Run all four retrieval configurations:

```bash
python scripts/run_baseline_comparison.py --device cpu
```

The experiment writes detailed reports under `examples/baseline/reports/`.
With the default BGE and CLIP models, the expected aggregate result is:

| System | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| `bge-text` | 0.6667 | 0.6667 | 0.6667 | 0.6667 |
| `bge-text-ocr` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `clip-visual` | 0.3333 | 1.0000 | 1.0000 | 0.6111 |
| `rrf-text-visual` | 0.6667 | 1.0000 | 1.0000 | 0.7778 |

The native system cannot retrieve the image-only microscopy page. Selective
OCR recovers that page while preserving retrieval performance on the
native-text pages. Generic CLIP retrieves every relevant page within the top
three, but its lower rank-one accuracy reflects the difficulty of reading small
scientific text from a resized full-page image. Equal-weight reciprocal rank
fusion preserves the native system's rank-one recall while improving its MRR
and top-three recall.

## OpenAI grounded answering

SightCite includes an optional OpenAI Responses API backend that sends only
retrieved page images to a vision-capable model and parses a structured answer
with explicit PDF page citations. The grounding service rejects citations to
pages outside the supplied evidence set and supports abstention when the
evidence is insufficient.

Install the optional backend:

```bash
python -m pip install --editable ".[openai]"
```

Set `OPENAI_API_KEY` in the environment before constructing
`OpenAIVisionLanguageModel`. Do not store API keys in source control. The
default model is `gpt-5.6-luna` and can be overridden in the constructor.

## End-to-end question answering

Install the optional OpenAI and LangChain dependencies:

```bash
python -m pip install --editable ".[openai,langchain]"
```

Set `OPENAI_API_KEY`, then answer a question using fused text and visual
retrieval, LangChain orchestration, and validated page citations:

```bash
sightcite answer paper.pdf \
  "What is the paper's main experimental result?" \
  --device cpu \
  --top-k 3
```

The command prints the grounded answer and the PDF page numbers supporting it.
It may download the configured embedding models on first use and makes a
billable OpenAI API request.

See all configuration options:

```bash
sightcite answer --help
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

Run the opt-in real-Tesseract smoke test:

```bash
SIGHTCITE_RUN_OCR_TESTS=1 pytest -m ocr -v --no-cov
```

Run the opt-in, billable OpenAI smoke test:

```bash
SIGHTCITE_RUN_OPENAI_TESTS=1 pytest -m openai -v --no-cov
```
