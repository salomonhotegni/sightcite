"""In-memory vector retrieval."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from sightcite.ingestion import RenderedPage, TextChunk
from sightcite.retrieval.models import SearchResult, VisualSearchResult


class _CosineVectorIndex:
    """Shared normalized cosine-similarity index."""

    def __init__(
        self,
        embeddings: npt.ArrayLike,
        *,
        item_count: int,
        item_name: str,
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float64)

        if matrix.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")

        if matrix.shape[1] == 0:
            raise ValueError("embeddings must have at least one dimension")

        if matrix.shape[0] != item_count:
            raise ValueError(f"the number of embeddings must match the number of {item_name}")

        if not np.all(np.isfinite(matrix)):
            raise ValueError("embeddings must contain only finite values")

        norms = np.linalg.norm(matrix, axis=1)

        if np.any(norms == 0):
            raise ValueError("embedding vectors must not be zero")

        self._embeddings = matrix / norms[:, np.newaxis]

    @property
    def size(self) -> int:
        """Return the number of indexed vectors."""
        return int(self._embeddings.shape[0])

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return int(self._embeddings.shape[1])

    def search(
        self,
        query_embedding: npt.ArrayLike,
        *,
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Return vector indices and cosine scores in ranked order."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query = np.asarray(
            query_embedding,
            dtype=np.float64,
        )

        if query.ndim != 1:
            raise ValueError("query_embedding must be one-dimensional")

        if query.shape[0] != self.dimension:
            raise ValueError("query embedding dimension does not match the index dimension")

        if not np.all(np.isfinite(query)):
            raise ValueError("query_embedding must contain only finite values")

        query_norm = float(np.linalg.norm(query))

        if query_norm == 0:
            raise ValueError("query_embedding must not be zero")

        scores = self._embeddings @ (query / query_norm)
        result_count = min(top_k, self.size)
        indices = np.argsort(
            -scores,
            kind="stable",
        )[:result_count]

        return [(int(index), float(scores[index])) for index in indices]


class TextVectorIndex:
    """An in-memory cosine-similarity index for text chunks."""

    def __init__(
        self,
        chunks: Sequence[TextChunk],
        embeddings: npt.ArrayLike,
    ) -> None:
        """Create an index from chunks and their vectors."""
        self._chunks = tuple(chunks)
        self._index = _CosineVectorIndex(
            embeddings,
            item_count=len(self._chunks),
            item_name="chunks",
        )

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return self._index.size

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._index.dimension

    def search(
        self,
        query_embedding: npt.ArrayLike,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return chunks ranked by cosine similarity."""
        matches = self._index.search(
            query_embedding,
            top_k=top_k,
        )

        return [
            SearchResult(
                rank=rank,
                score=score,
                chunk=self._chunks[index],
            )
            for rank, (index, score) in enumerate(
                matches,
                start=1,
            )
        ]


class VisualVectorIndex:
    """An in-memory cosine-similarity index for rendered pages."""

    def __init__(
        self,
        pages: Sequence[RenderedPage],
        embeddings: npt.ArrayLike,
    ) -> None:
        """Create an index from rendered pages and their vectors."""
        self._pages = tuple(pages)
        self._index = _CosineVectorIndex(
            embeddings,
            item_count=len(self._pages),
            item_name="pages",
        )

    @property
    def size(self) -> int:
        """Return the number of indexed pages."""
        return self._index.size

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._index.dimension

    def search(
        self,
        query_embedding: npt.ArrayLike,
        *,
        top_k: int = 5,
    ) -> list[VisualSearchResult]:
        """Return pages ranked by cosine similarity."""
        matches = self._index.search(
            query_embedding,
            top_k=top_k,
        )

        return [
            VisualSearchResult(
                rank=rank,
                score=score,
                page=self._pages[index],
            )
            for rank, (index, score) in enumerate(
                matches,
                start=1,
            )
        ]
