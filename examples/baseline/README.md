# Synthetic baseline benchmark

This fixture evaluates page retrieval on a controlled three-page scientific
document:

1. A native-text page about photovoltaic efficiency.
2. An image-only page about graphene microscopy.
3. A native-text page about battery capacity retention.

`benchmark.json` contains one page-level retrieval question per page.

Generate the PDF:

```bash
python scripts/generate_baseline_fixture.py
```

Run the native-text, OCR-assisted, CLIP visual, and fused baselines:

```bash
python scripts/run_baseline_comparison.py --device cpu
```

The PDF and generated reports are intentionally ignored by Git because they
are reproducible artifacts.

The comparison writes individual JSON reports for `bge-text`, `bge-text-ocr`,
`clip-visual`, and `rrf-text-visual`, plus `comparison.json` containing
aggregate metrics, fusion configuration, and deltas relative to the
native-text baseline.
