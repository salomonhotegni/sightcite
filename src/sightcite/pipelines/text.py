"""End-to-end text retrieval pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory

from sightcite.ingestion import (
    ExtractedPage,
    OcrBackend,
    TextChunk,
    chunk_extracted_pages,
    extract_pdf_text,
    extract_pdf_text_with_ocr,
)
from sightcite.retrieval import SearchResult, TextEmbedder, TextRetriever


class TextRetrievalPipeline:
    """Index and search text extracted from one PDF."""

    def __init__(
        self,
        pdf_path: str | Path,
        embedder: TextEmbedder,
        *,
        chunk_size: int = 200,
        overlap: int = 40,
        ocr_backend: OcrBackend | None = None,
        ocr_output_dir: str | Path | None = None,
        min_native_chars: int = 20,
        ocr_dpi: int = 144,
    ) -> None:
        source = Path(pdf_path)

        if ocr_backend is None:
            pages = extract_pdf_text(source)
        elif ocr_output_dir is not None:
            pages = extract_pdf_text_with_ocr(
                source,
                ocr_output_dir,
                ocr_backend,
                min_native_chars=min_native_chars,
                dpi=ocr_dpi,
            )
        else:
            with TemporaryDirectory(prefix="sightcite-ocr-") as temporary_dir:
                pages = extract_pdf_text_with_ocr(
                    source,
                    temporary_dir,
                    ocr_backend,
                    min_native_chars=min_native_chars,
                    dpi=ocr_dpi,
                )

        chunks = chunk_extracted_pages(
            pages,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not chunks:
            raise ValueError("PDF contains no extractable text")

        self._source = source
        self._pages = tuple(pages)
        self._chunks = tuple(chunks)
        self._retriever = TextRetriever(self._chunks, embedder)

    @property
    def source(self) -> Path:
        """Return the indexed PDF path."""
        return self._source

    @property
    def pages(self) -> tuple[ExtractedPage, ...]:
        """Return the extracted pages."""
        return self._pages

    @property
    def chunks(self) -> tuple[TextChunk, ...]:
        """Return the indexed text chunks."""
        return self._chunks

    @property
    def page_count(self) -> int:
        """Return the number of PDF pages."""
        return len(self._pages)

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return chunks ranked for a natural-language query."""
        return self._retriever.search(query, top_k=top_k)
