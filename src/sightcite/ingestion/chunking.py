"""Text chunking for retrieval."""

from collections.abc import Sequence

from sightcite.ingestion.models import ExtractedPage, TextChunk


def chunk_extracted_pages(
    pages: Sequence[ExtractedPage],
    *,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[TextChunk]:
    """Split extracted page text into overlapping word-based chunks.

    Args:
        pages: Extracted PDF pages in document order.
        chunk_size: Maximum number of words in each chunk.
        overlap: Number of words shared by consecutive chunks.

    Returns:
        Retrieval-ready chunks. Blank pages produce no chunks.

    Raises:
        ValueError: If the chunk size or overlap is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap must not be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks: list[TextChunk] = []

    for page in pages:
        words = page.text.split()

        for chunk_index, start_word in enumerate(range(0, len(words), step)):
            end_word = min(start_word + chunk_size, len(words))

            chunks.append(
                TextChunk(
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=" ".join(words[start_word:end_word]),
                    start_word=start_word,
                    end_word=end_word,
                )
            )

            if end_word == len(words):
                break

    return chunks
