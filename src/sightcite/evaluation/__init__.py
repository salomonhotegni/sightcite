"""SightCite evaluation utilities."""

from sightcite.evaluation.benchmark import run_text_retrieval_benchmark
from sightcite.evaluation.dataset import load_retrieval_benchmark
from sightcite.evaluation.models import (
    BenchmarkResult,
    QueryRetrievalEvaluation,
    RetrievalBenchmark,
    RetrievalEvaluation,
    RetrievalExample,
    RetrievalMetrics,
)
from sightcite.evaluation.report import write_benchmark_report
from sightcite.evaluation.retrieval import (
    PageRankingFunction,
    evaluate_retrieval,
)

__all__ = [
    "BenchmarkResult",
    "PageRankingFunction",
    "QueryRetrievalEvaluation",
    "RetrievalBenchmark",
    "RetrievalEvaluation",
    "RetrievalExample",
    "RetrievalMetrics",
    "evaluate_retrieval",
    "load_retrieval_benchmark",
    "run_text_retrieval_benchmark",
    "write_benchmark_report",
]
