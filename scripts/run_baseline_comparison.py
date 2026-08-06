"""Run SightCite's text, OCR, and visual retrieval baselines."""

import argparse
import json
from pathlib import Path

from sightcite.evaluation import (
    BenchmarkResult,
    load_retrieval_benchmark,
    run_text_retrieval_benchmark,
    run_visual_retrieval_benchmark,
    write_benchmark_report,
)
from sightcite.ingestion import TesseractOcrBackend
from sightcite.retrieval import (
    DEFAULT_BGE_MODEL,
    DEFAULT_CLIP_MODEL,
    BgeTextEmbedder,
    ClipVisualEmbedder,
)

DEFAULT_DATASET = Path("examples/baseline/benchmark.json")
DEFAULT_OUTPUT_DIR = Path("examples/baseline/reports")
METRIC_NAMES = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "mean_reciprocal_rank",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the baseline comparison argument parser."""
    parser = argparse.ArgumentParser(
        description=("Compare native-text, OCR-assisted, and visual retrieval."),
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
        "--text-model",
        "--model",
        dest="text_model",
        default=DEFAULT_BGE_MODEL,
        help="Sentence Transformers text model name.",
    )
    parser.add_argument(
        "--visual-model",
        default=DEFAULT_CLIP_MODEL,
        help="Sentence Transformers visual model name.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device such as cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=32,
        help="Text embedding batch size.",
    )
    parser.add_argument(
        "--visual-batch-size",
        type=int,
        default=16,
        help="Visual embedding batch size.",
    )
    parser.add_argument(
        "--visual-dpi",
        type=int,
        default=144,
        help="PDF rendering resolution for visual retrieval.",
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


def _metric_deltas(
    baseline: dict[str, int | float],
    candidate: dict[str, int | float],
) -> dict[str, int | float]:
    return {metric: candidate[metric] - baseline[metric] for metric in METRIC_NAMES}


def _print_metrics(
    system_name: str,
    metrics: dict[str, int | float],
) -> None:
    print(
        f"{system_name:<15}"
        f"{metrics['recall_at_1']:>10.4f}"
        f"{metrics['recall_at_3']:>10.4f}"
        f"{metrics['recall_at_5']:>10.4f}"
        f"{metrics['mean_reciprocal_rank']:>10.4f}"
    )


def main() -> None:
    """Run all retrieval modes and write their comparison."""
    arguments = build_parser().parse_args()
    dataset_path: Path = arguments.dataset
    output_dir: Path = arguments.output_dir
    text_model: str = arguments.text_model
    visual_model: str = arguments.visual_model
    device: str | None = arguments.device
    text_batch_size: int = arguments.text_batch_size
    visual_batch_size: int = arguments.visual_batch_size
    visual_dpi: int = arguments.visual_dpi

    if text_batch_size <= 0:
        raise ValueError("text-batch-size must be greater than zero")

    if visual_batch_size <= 0:
        raise ValueError("visual-batch-size must be greater than zero")

    if visual_dpi <= 0:
        raise ValueError("visual-dpi must be greater than zero")

    benchmark = load_retrieval_benchmark(dataset_path)

    text_embedder = BgeTextEmbedder(
        text_model,
        device=device,
        batch_size=text_batch_size,
    )
    native_result = run_text_retrieval_benchmark(
        benchmark,
        text_embedder,
        system_name="bge-text",
    )
    ocr_result = run_text_retrieval_benchmark(
        benchmark,
        text_embedder,
        system_name="bge-text-ocr",
        ocr_backend=TesseractOcrBackend(),
    )

    visual_embedder = ClipVisualEmbedder(
        visual_model,
        device=device,
        batch_size=visual_batch_size,
    )
    visual_result = run_visual_retrieval_benchmark(
        benchmark,
        visual_embedder,
        system_name="clip-visual",
        dpi=visual_dpi,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    report_paths = {
        "bge-text": write_benchmark_report(
            native_result,
            output_dir / "bge-text.json",
        ),
        "bge-text-ocr": write_benchmark_report(
            ocr_result,
            output_dir / "bge-text-ocr.json",
        ),
        "clip-visual": write_benchmark_report(
            visual_result,
            output_dir / "clip-visual.json",
        ),
    }

    native_metrics = _metrics_payload(native_result)
    ocr_metrics = _metrics_payload(ocr_result)
    visual_metrics = _metrics_payload(visual_result)

    comparison = {
        "schema_version": 1,
        "document_id": benchmark.document_id,
        "systems": {
            "bge-text": native_metrics,
            "bge-text-ocr": ocr_metrics,
            "clip-visual": visual_metrics,
        },
        "ocr_minus_native": _metric_deltas(
            native_metrics,
            ocr_metrics,
        ),
        "visual_minus_native": _metric_deltas(
            native_metrics,
            visual_metrics,
        ),
    }

    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            comparison,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for system_name, report_path in report_paths.items():
        print(f"{system_name}: {report_path}")

    print(f"Comparison: {comparison_path}")
    print()
    print(f"{'System':<15}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}{'MRR':>10}")
    _print_metrics("bge-text", native_metrics)
    _print_metrics("bge-text-ocr", ocr_metrics)
    _print_metrics("clip-visual", visual_metrics)


if __name__ == "__main__":
    main()
