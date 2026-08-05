"""End-to-end retrieval benchmark execution."""

from sightcite.evaluation.models import (
    BenchmarkResult,
    RetrievalBenchmark,
)
from sightcite.evaluation.retrieval import evaluate_retrieval
from sightcite.pipelines import TextRetrievalPipeline
from sightcite.retrieval import TextEmbedder


def run_text_retrieval_benchmark(
    benchmark: RetrievalBenchmark,
    embedder: TextEmbedder,
    *,
    system_name: str = "bge-text",
    chunk_size: int = 200,
    overlap: int = 40,
) -> BenchmarkResult:
    """Run a page-retrieval benchmark with the native-text pipeline."""
    pipeline = TextRetrievalPipeline(
        benchmark.pdf_path,
        embedder,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    def rank_pages(query: str) -> list[int]:
        results = pipeline.search(
            query,
            top_k=pipeline.chunk_count,
        )
        return [result.chunk.page_number for result in results]

    evaluation = evaluate_retrieval(
        benchmark.examples,
        rank_pages,
    )

    return BenchmarkResult(
        system_name=system_name,
        document_id=benchmark.document_id,
        pdf_path=benchmark.pdf_path,
        evaluation=evaluation,
    )
