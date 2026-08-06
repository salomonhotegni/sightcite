"""High-level visual page retrieval."""

from collections.abc import Sequence

from sightcite.ingestion import RenderedPage
from sightcite.retrieval.index import VisualVectorIndex
from sightcite.retrieval.models import VisualSearchResult
from sightcite.retrieval.visual_embeddings import VisualEmbedder


class VisualRetriever:
    """Embed, index, and retrieve rendered PDF pages."""

    def __init__(
        self,
        pages: Sequence[RenderedPage],
        embedder: VisualEmbedder,
    ) -> None:
        self._embedder = embedder
        embeddings = embedder.embed_images([page.image_path for page in pages])
        self._index = VisualVectorIndex(
            pages,
            embeddings,
        )

    @property
    def size(self) -> int:
        """Return the number of indexed pages."""
        return self._index.size

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[VisualSearchResult]:
        """Retrieve pages relevant to a natural-language query."""
        query_embedding = self._embedder.embed_query(query)

        return self._index.search(
            query_embedding,
            top_k=top_k,
        )
