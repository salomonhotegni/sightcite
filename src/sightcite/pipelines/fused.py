"""Unified text and visual retrieval pipeline."""

from pathlib import Path
from types import TracebackType

from sightcite.ingestion import (
    ExtractedPage,
    OcrBackend,
    RenderedPage,
    TextChunk,
)
from sightcite.pipelines.text import TextRetrievalPipeline
from sightcite.pipelines.visual import VisualRetrievalPipeline
from sightcite.retrieval import (
    DEFAULT_RRF_CONSTANT,
    FusedSearchResult,
    TextEmbedder,
    VisualEmbedder,
    reciprocal_rank_fusion,
)


class FusedRetrievalPipeline:
    """Retrieve PDF pages using fused text and visual rankings."""

    def __init__(
        self,
        pdf_path: str | Path,
        text_embedder: TextEmbedder,
        visual_embedder: VisualEmbedder,
        *,
        chunk_size: int = 200,
        overlap: int = 40,
        ocr_backend: OcrBackend | None = None,
        ocr_output_dir: str | Path | None = None,
        min_native_chars: int = 20,
        ocr_dpi: int = 144,
        visual_output_dir: str | Path | None = None,
        visual_dpi: int = 144,
        rank_constant: int = DEFAULT_RRF_CONSTANT,
        text_weight: float = 1.0,
        visual_weight: float = 1.0,
    ) -> None:
        if rank_constant < 0:
            raise ValueError("rank_constant must not be negative")

        if text_weight <= 0:
            raise ValueError("text_weight must be greater than zero")

        if visual_weight <= 0:
            raise ValueError("visual_weight must be greater than zero")

        text_pipeline = TextRetrievalPipeline(
            pdf_path,
            text_embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            ocr_backend=ocr_backend,
            ocr_output_dir=ocr_output_dir,
            min_native_chars=min_native_chars,
            ocr_dpi=ocr_dpi,
        )
        visual_pipeline = VisualRetrievalPipeline(
            pdf_path,
            visual_embedder,
            output_dir=visual_output_dir,
            dpi=visual_dpi,
        )

        self._source = Path(pdf_path)
        self._text_pipeline = text_pipeline
        self._visual_pipeline = visual_pipeline
        self._pages_by_number = {page.page_number: page for page in visual_pipeline.pages}
        self._rank_constant = rank_constant
        self._weights = {
            "text": text_weight,
            "visual": visual_weight,
        }

    @property
    def source(self) -> Path:
        """Return the indexed PDF path."""
        return self._source

    @property
    def pages(self) -> tuple[RenderedPage, ...]:
        """Return the rendered evidence pages."""
        return self._visual_pipeline.pages

    @property
    def text_pages(self) -> tuple[ExtractedPage, ...]:
        """Return the extracted text pages."""
        return self._text_pipeline.pages

    @property
    def chunks(self) -> tuple[TextChunk, ...]:
        """Return the indexed text chunks."""
        return self._text_pipeline.chunks

    @property
    def page_count(self) -> int:
        """Return the number of indexed PDF pages."""
        return self._visual_pipeline.page_count

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[FusedSearchResult]:
        """Return pages ranked by fused text and visual retrieval."""
        text_results = self._text_pipeline.search(
            query,
            top_k=self._text_pipeline.chunk_count,
        )
        visual_results = self._visual_pipeline.search(
            query,
            top_k=self._visual_pipeline.page_count,
        )

        fused_results = reciprocal_rank_fusion(
            {
                "text": [result.chunk.page_number for result in text_results],
                "visual": [result.page.page_number for result in visual_results],
            },
            rank_constant=self._rank_constant,
            weights=self._weights,
            top_k=top_k,
        )

        return [
            FusedSearchResult(
                rank=result.rank,
                score=result.score,
                page=self._pages_by_number[result.page_number],
                source_ranks=result.source_ranks,
            )
            for result in fused_results
        ]

    def close(self) -> None:
        """Release temporary rendered page images."""
        self._visual_pipeline.close()

    def __enter__(self) -> "FusedRetrievalPipeline":
        """Enter the pipeline resource context."""
        self._visual_pipeline.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release temporary resources on context exit."""
        self.close()
