"""SightCite command-line interface."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from sightcite.evaluation import (
    load_retrieval_benchmark,
    run_text_retrieval_benchmark,
    write_benchmark_report,
)
from sightcite.retrieval import DEFAULT_BGE_MODEL, BgeTextEmbedder


def build_parser() -> argparse.ArgumentParser:
    """Create the SightCite argument parser."""
    parser = argparse.ArgumentParser(
        prog="sightcite",
        description="Visual and textual retrieval for scientific papers.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run a page-retrieval benchmark.",
    )
    benchmark_parser.add_argument(
        "dataset",
        help="Path to the benchmark JSON dataset.",
    )
    benchmark_parser.add_argument(
        "--output",
        required=True,
        help="Path for the JSON benchmark report.",
    )
    benchmark_parser.add_argument(
        "--model",
        default=DEFAULT_BGE_MODEL,
        help="Sentence Transformers model name.",
    )
    benchmark_parser.add_argument(
        "--device",
        default=None,
        help="Inference device, such as cpu, cuda, or cuda:0.",
    )
    benchmark_parser.add_argument(
        "--system-name",
        default="bge-text",
        help="System name recorded in the report.",
    )
    benchmark_parser.add_argument(
        "--chunk-size",
        type=_positive_integer,
        default=200,
        help="Maximum words per text chunk.",
    )
    benchmark_parser.add_argument(
        "--overlap",
        type=_non_negative_integer,
        default=40,
        help="Words shared by consecutive chunks.",
    )
    benchmark_parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=32,
        help="Embedding batch size.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SightCite command-line interface."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    command = cast(str, arguments.command)

    try:
        if command == "benchmark":
            return _run_benchmark(arguments)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    parser.error(f"unknown command: {command}")


def _run_benchmark(arguments: argparse.Namespace) -> int:
    dataset_path = Path(cast(str, arguments.dataset))
    output_path = Path(cast(str, arguments.output))
    model_name = cast(str, arguments.model)
    device = cast(str | None, arguments.device)
    system_name = cast(str, arguments.system_name)
    chunk_size = cast(int, arguments.chunk_size)
    overlap = cast(int, arguments.overlap)
    batch_size = cast(int, arguments.batch_size)

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk-size")

    benchmark = load_retrieval_benchmark(dataset_path)
    embedder = BgeTextEmbedder(
        model_name,
        device=device,
        batch_size=batch_size,
    )
    result = run_text_retrieval_benchmark(
        benchmark,
        embedder,
        system_name=system_name,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    report_path = write_benchmark_report(result, output_path)
    metrics = result.evaluation.metrics

    print(f"Document: {result.document_id}")
    print(f"Queries: {metrics.query_count}")
    print(f"Recall@1: {metrics.recall_at_1:.4f}")
    print(f"Recall@3: {metrics.recall_at_3:.4f}")
    print(f"Recall@5: {metrics.recall_at_5:.4f}")
    print(f"MRR: {metrics.mean_reciprocal_rank:.4f}")
    print(f"Report: {report_path}")

    return 0


def _positive_integer(value: str) -> int:
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")

    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)

    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")

    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
