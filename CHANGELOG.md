# Changelog

All notable changes to SightCite are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-08-09

### Added

- PDF page rendering with PyMuPDF.
- Native page-level text extraction.
- Selective Tesseract OCR for scanned or text-poor pages.
- Page-aware overlapping text chunking.
- BGE text embeddings and cosine-similarity retrieval.
- CLIP page-image embeddings and visual retrieval.
- Reciprocal rank fusion of text and visual rankings.
- End-to-end text, visual, and fused retrieval pipelines.
- Retrieval evaluation with Recall@1, Recall@3, Recall@5, and mean reciprocal
  rank.
- Versioned benchmark datasets and JSON reports.
- Reproducible synthetic scientific-paper baseline.
- Vision-language-model interfaces for structured, cited answers.
- Citation validation and insufficient-evidence abstention.
- OpenAI Responses API vision backend.
- LangChain runnable orchestration for retrieval and grounded answering.
- `sightcite benchmark` command for retrieval evaluation.
- `sightcite answer` command for end-to-end question answering.
- Optional OpenAI, LangChain, and release-tool dependency groups.
- GitHub Actions checks on Python 3.11 and 3.12.
- MIT licensing and PEP 639 package metadata.

### Quality

- Strict mypy type checking.
- Ruff linting and formatting.
- More than 180 automated tests.
- More than 97 percent test coverage.
- Opt-in integration tests for model inference, Tesseract OCR, and OpenAI.
- Validated wheel and source distribution artifacts.

[Unreleased]: https://github.com/salomonhotegni/sightcite/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/salomonhotegni/sightcite/releases/tag/v0.1.0