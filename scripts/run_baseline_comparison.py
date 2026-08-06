"""Run the native-text and OCR-assisted SightCite baselines."""

import argparse
import json
from pathlib import Path

from sightcite.evaluation import (
    BenchmarkResult,
    load_retrieval_benchmark,
    run_text_retrieval_benchmark,
    write_benchmark_report,
)
from sightcite.ingestion import TesseractOcrBackend
from sightcite.retrieval import DEFAULT_BGE_MODEL, BgeTextEmbedder

DEFAULT_DATASET = Path("examples/baseline/benchmark.json")
DEFAULT_OUTPUT_DIR = Path("examples/baseline/reports")


def build_parser() -> argparse.ArgumentParser:
    """Create the baseline comparison argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare native-text and OCR-assisted retrieval.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the benchmark dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark reports.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BGE_MODEL,
        help="Sentence Transformers model name.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device such as cpu, cuda, or cuda:0.",
    )
    return parser


def _metrics_payload(
    result: BenchmarkResult,
) -> dict[str, int | float]:
    metrics = result.evaluation.metrics

    return {
        "query_count": metrics.query_count,
        "recall_at_1": metrics.recall_at_1,
        "recall_at_3": metrics.recall_at_3,
        "recall_at_5": metrics.recall_at_5,
        "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
    }


def main() -> None:
    """Run both retrieval modes and write their comparison."""
    arguments = build_parser().parse_args()
    dataset_path: Path = arguments.dataset
    output_dir: Path = arguments.output_dir
    model_name: str = arguments.model
    device: str | None = arguments.device

    benchmark = load_retrieval_benchmark(dataset_path)
    embedder = BgeTextEmbedder(
        model_name,
        device=device,
    )

    native_result = run_text_retrieval_benchmark(
        benchmark,
        embedder,
        system_name="bge-text",
    )
    ocr_result = run_text_retrieval_benchmark(
        benchmark,
        embedder,
        system_name="bge-text-ocr",
        ocr_backend=TesseractOcrBackend(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    native_report = write_benchmark_report(
        native_result,
        output_dir / "bge-text.json",
    )
    ocr_report = write_benchmark_report(
        ocr_result,
        output_dir / "bge-text-ocr.json",
    )

    native_metrics = _metrics_payload(native_result)
    ocr_metrics = _metrics_payload(ocr_result)

    comparison = {
        "schema_version": 1,
        "document_id": benchmark.document_id,
        "systems": {
            "bge-text": native_metrics,
            "bge-text-ocr": ocr_metrics,
        },
        "ocr_minus_native": {
            metric: ocr_metrics[metric] - native_metrics[metric]
            for metric in (
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "mean_reciprocal_rank",
            )
        },
    }

    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Native report: {native_report}")
    print(f"OCR report: {ocr_report}")
    print(f"Comparison: {comparison_path}")
    print()
    print("System        Recall@1  Recall@3  Recall@5  MRR")
    print(
        f"bge-text      {native_metrics['recall_at_1']:.4f}    "
        f"{native_metrics['recall_at_3']:.4f}    "
        f"{native_metrics['recall_at_5']:.4f}    "
        f"{native_metrics['mean_reciprocal_rank']:.4f}"
    )
    print(
        f"bge-text-ocr  {ocr_metrics['recall_at_1']:.4f}    "
        f"{ocr_metrics['recall_at_3']:.4f}    "
        f"{ocr_metrics['recall_at_5']:.4f}    "
        f"{ocr_metrics['mean_reciprocal_rank']:.4f}"
    )


if __name__ == "__main__":
    main()
