"""End-to-end retrieval benchmark execution."""

from pathlib import Path

from sightcite.evaluation.models import (
    BenchmarkResult,
    RetrievalBenchmark,
)
from sightcite.evaluation.retrieval import evaluate_retrieval
from sightcite.ingestion import OcrBackend
from sightcite.pipelines import (
    TextRetrievalPipeline,
    VisualRetrievalPipeline,
)
from sightcite.retrieval import TextEmbedder, VisualEmbedder


def run_text_retrieval_benchmark(
    benchmark: RetrievalBenchmark,
    embedder: TextEmbedder,
    *,
    system_name: str = "bge-text",
    chunk_size: int = 200,
    overlap: int = 40,
    ocr_backend: OcrBackend | None = None,
    ocr_output_dir: str | Path | None = None,
    min_native_chars: int = 20,
    ocr_dpi: int = 144,
) -> BenchmarkResult:
    """Run a page-retrieval benchmark with the text pipeline."""
    pipeline = TextRetrievalPipeline(
        benchmark.pdf_path,
        embedder,
        chunk_size=chunk_size,
        overlap=overlap,
        ocr_backend=ocr_backend,
        ocr_output_dir=ocr_output_dir,
        min_native_chars=min_native_chars,
        ocr_dpi=ocr_dpi,
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


def run_visual_retrieval_benchmark(
    benchmark: RetrievalBenchmark,
    embedder: VisualEmbedder,
    *,
    system_name: str = "clip-visual",
    output_dir: str | Path | None = None,
    dpi: int = 144,
) -> BenchmarkResult:
    """Run a page-retrieval benchmark with the visual pipeline."""
    with VisualRetrievalPipeline(
        benchmark.pdf_path,
        embedder,
        output_dir=output_dir,
        dpi=dpi,
    ) as pipeline:

        def rank_pages(query: str) -> list[int]:
            results = pipeline.search(
                query,
                top_k=pipeline.page_count,
            )
            return [result.page.page_number for result in results]

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
