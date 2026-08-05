"""End-to-end native-text retrieval pipeline."""

from pathlib import Path

from sightcite.ingestion import (
    ExtractedPage,
    TextChunk,
    chunk_extracted_pages,
    extract_pdf_text,
)
from sightcite.retrieval import SearchResult, TextEmbedder, TextRetriever


class TextRetrievalPipeline:
    """Index and search the native text embedded in one PDF."""

    def __init__(
        self,
        pdf_path: str | Path,
        embedder: TextEmbedder,
        *,
        chunk_size: int = 200,
        overlap: int = 40,
    ) -> None:
        source = Path(pdf_path)
        pages = extract_pdf_text(source)
        chunks = chunk_extracted_pages(
            pages,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not chunks:
            raise ValueError("PDF contains no extractable native text; OCR may be required")

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
