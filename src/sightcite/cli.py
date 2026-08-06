"""SightCite command-line interface."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from sightcite.evaluation import (
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
        default=None,
        help=(
            "Embedding model name. Defaults to the BGE or CLIP "
            "baseline according to retrieval mode."
        ),
    )
    benchmark_parser.add_argument(
        "--device",
        default=None,
        help="Inference device, such as cpu, cuda, or cuda:0.",
    )
    benchmark_parser.add_argument(
        "--system-name",
        default=None,
        help=(
            "System name recorded in the report. Defaults to bge-text, "
            "bge-text-ocr, or clip-visual according to retrieval mode."
        ),
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
        default=None,
        help=("Embedding batch size. Defaults to 32 for text and 16 for visual retrieval."),
    )
    retrieval_mode = benchmark_parser.add_mutually_exclusive_group()
    retrieval_mode.add_argument(
        "--ocr",
        action="store_true",
        help="Apply Tesseract OCR to pages with insufficient native text.",
    )
    retrieval_mode.add_argument(
        "--visual",
        action="store_true",
        help="Retrieve rendered pages using CLIP visual embeddings.",
    )
    benchmark_parser.add_argument(
        "--visual-dpi",
        type=_positive_integer,
        default=144,
        help="PDF rendering resolution for visual retrieval.",
    )
    benchmark_parser.add_argument(
        "--visual-output-dir",
        default=None,
        help="Optional directory in which rendered visual pages are retained.",
    )
    benchmark_parser.add_argument(
        "--ocr-language",
        default="eng",
        help="Tesseract language code.",
    )
    benchmark_parser.add_argument(
        "--ocr-page-segmentation-mode",
        type=_page_segmentation_mode,
        default=3,
        help="Tesseract page segmentation mode from 0 through 13.",
    )
    benchmark_parser.add_argument(
        "--ocr-timeout",
        type=_positive_float,
        default=30.0,
        help="Maximum OCR processing time per page in seconds.",
    )
    benchmark_parser.add_argument(
        "--ocr-min-native-chars",
        type=_non_negative_integer,
        default=20,
        help="Minimum non-whitespace native characters required to skip OCR.",
    )
    benchmark_parser.add_argument(
        "--ocr-dpi",
        type=_positive_integer,
        default=144,
        help="Rendering resolution used for OCR.",
    )
    benchmark_parser.add_argument(
        "--ocr-output-dir",
        default=None,
        help="Optional directory in which rendered OCR page images are retained.",
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
    requested_model_name = cast(str | None, arguments.model)
    device = cast(str | None, arguments.device)
    requested_system_name = cast(str | None, arguments.system_name)
    chunk_size = cast(int, arguments.chunk_size)
    overlap = cast(int, arguments.overlap)
    requested_batch_size = cast(int | None, arguments.batch_size)
    use_ocr = cast(bool, arguments.ocr)
    use_visual = cast(bool, arguments.visual)
    ocr_language = cast(str, arguments.ocr_language)
    ocr_page_segmentation_mode = cast(
        int,
        arguments.ocr_page_segmentation_mode,
    )
    ocr_timeout = cast(float, arguments.ocr_timeout)
    ocr_min_native_chars = cast(int, arguments.ocr_min_native_chars)
    ocr_dpi = cast(int, arguments.ocr_dpi)
    ocr_output_dir_value = cast(str | None, arguments.ocr_output_dir)
    visual_dpi = cast(int, arguments.visual_dpi)
    visual_output_dir_value = cast(
        str | None,
        arguments.visual_output_dir,
    )

    if requested_system_name is not None:
        system_name = requested_system_name
    elif use_visual:
        system_name = "clip-visual"
    elif use_ocr:
        system_name = "bge-text-ocr"
    else:
        system_name = "bge-text"
    model_name = requested_model_name or (DEFAULT_CLIP_MODEL if use_visual else DEFAULT_BGE_MODEL)
    batch_size = requested_batch_size or (16 if use_visual else 32)
    ocr_output_dir = Path(ocr_output_dir_value) if ocr_output_dir_value is not None else None
    visual_output_dir = (
        Path(visual_output_dir_value) if visual_output_dir_value is not None else None
    )
    ocr_backend = (
        TesseractOcrBackend(
            language=ocr_language,
            page_segmentation_mode=ocr_page_segmentation_mode,
            timeout_seconds=ocr_timeout,
        )
        if use_ocr
        else None
    )

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk-size")

    benchmark = load_retrieval_benchmark(dataset_path)
    if use_visual:
        visual_embedder = ClipVisualEmbedder(
            model_name,
            device=device,
            batch_size=batch_size,
        )
        result = run_visual_retrieval_benchmark(
            benchmark,
            visual_embedder,
            system_name=system_name,
            output_dir=visual_output_dir,
            dpi=visual_dpi,
        )
    else:
        text_embedder = BgeTextEmbedder(
            model_name,
            device=device,
            batch_size=batch_size,
        )
        result = run_text_retrieval_benchmark(
            benchmark,
            text_embedder,
            system_name=system_name,
            chunk_size=chunk_size,
            overlap=overlap,
            ocr_backend=ocr_backend,
            ocr_output_dir=ocr_output_dir,
            min_native_chars=ocr_min_native_chars,
            ocr_dpi=ocr_dpi,
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


def _positive_float(value: str) -> float:
    parsed = float(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")

    return parsed


def _page_segmentation_mode(value: str) -> int:
    parsed = int(value)

    if not 0 <= parsed <= 13:
        raise argparse.ArgumentTypeError("value must be between 0 and 13")

    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
