"""End-to-end visual page retrieval pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType

from sightcite.ingestion import RenderedPage, render_pdf
from sightcite.retrieval import (
    VisualEmbedder,
    VisualRetriever,
    VisualSearchResult,
)


class VisualRetrievalPipeline:
    """Render, index, and search the pages of one PDF."""

    def __init__(
        self,
        pdf_path: str | Path,
        embedder: VisualEmbedder,
        *,
        output_dir: str | Path | None = None,
        dpi: int = 144,
    ) -> None:
        source = Path(pdf_path)
        temporary_directory: TemporaryDirectory[str] | None = None

        if output_dir is None:
            temporary_directory = TemporaryDirectory(prefix="sightcite-visual-")
            rendered_output_dir = Path(temporary_directory.name)
        else:
            rendered_output_dir = Path(output_dir)

        try:
            pages = render_pdf(
                source,
                rendered_output_dir,
                dpi=dpi,
            )
            retriever = VisualRetriever(
                pages,
                embedder,
            )
        except Exception:
            if temporary_directory is not None:
                temporary_directory.cleanup()
            raise

        self._source = source
        self._pages = tuple(pages)
        self._retriever = retriever
        self._temporary_directory = temporary_directory
        self._closed = False

    @property
    def source(self) -> Path:
        """Return the indexed PDF path."""
        return self._source

    @property
    def pages(self) -> tuple[RenderedPage, ...]:
        """Return the rendered pages."""
        return self._pages

    @property
    def page_count(self) -> int:
        """Return the number of indexed pages."""
        return len(self._pages)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[VisualSearchResult]:
        """Return pages ranked for a natural-language query."""
        self._ensure_open()
        return self._retriever.search(
            query,
            top_k=top_k,
        )

    def close(self) -> None:
        """Release temporary rendered page images."""
        if self._closed:
            return

        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()

        self._closed = True

    def __enter__(self) -> "VisualRetrievalPipeline":
        """Enter the pipeline resource context."""
        self._ensure_open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release temporary resources on context exit."""
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("visual retrieval pipeline is closed")
