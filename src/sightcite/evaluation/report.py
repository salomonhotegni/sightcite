"""Benchmark report serialization."""

import json
from pathlib import Path

from sightcite.evaluation.models import BenchmarkResult


def write_benchmark_report(
    result: BenchmarkResult,
    output_path: str | Path,
) -> Path:
    """Write a benchmark result as human-readable JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    metrics = result.evaluation.metrics

    payload = {
        "schema_version": 1,
        "system_name": result.system_name,
        "document_id": result.document_id,
        "pdf_path": str(result.pdf_path),
        "metrics": {
            "query_count": metrics.query_count,
            "recall_at_1": metrics.recall_at_1,
            "recall_at_3": metrics.recall_at_3,
            "recall_at_5": metrics.recall_at_5,
            "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
        },
        "queries": [
            {
                "query_id": query.query_id,
                "query": query.query,
                "relevant_pages": sorted(query.relevant_pages),
                "retrieved_pages": list(query.retrieved_pages),
                "recall_at_1": query.recall_at_1,
                "recall_at_3": query.recall_at_3,
                "recall_at_5": query.recall_at_5,
                "reciprocal_rank": query.reciprocal_rank,
            }
            for query in result.evaluation.queries
        ],
    }

    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return destination
