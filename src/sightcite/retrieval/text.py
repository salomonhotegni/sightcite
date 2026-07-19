"""High-level text retrieval."""

from collections.abc import Sequence

from sightcite.ingestion import TextChunk
from sightcite.retrieval.embeddings import TextEmbedder
from sightcite.retrieval.index import TextVectorIndex
from sightcite.retrieval.models import SearchResult


class TextRetriever:
    """Embed, index, and retrieve text chunks."""

    def __init__(
        self,
        chunks: Sequence[TextChunk],
        embedder: TextEmbedder,
    ) -> None:
        self._embedder = embedder
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
        self._index = TextVectorIndex(chunks, embeddings)

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return self._index.size

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Retrieve chunks relevant to a natural-language query."""
        query_embedding = self._embedder.embed_query(query)

        return self._index.search(query_embedding, top_k=top_k)
