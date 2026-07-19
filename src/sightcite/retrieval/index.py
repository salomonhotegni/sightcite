"""In-memory vector retrieval."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from sightcite.ingestion import TextChunk
from sightcite.retrieval.models import SearchResult


class TextVectorIndex:
    """An in-memory cosine-similarity index for text chunks."""

    def __init__(
        self,
        chunks: Sequence[TextChunk],
        embeddings: npt.ArrayLike,
    ) -> None:
        """Create an index from chunks and their embedding vectors."""
        matrix = np.asarray(embeddings, dtype=np.float64)

        if matrix.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")

        if matrix.shape[1] == 0:
            raise ValueError("embeddings must have at least one dimension")

        if matrix.shape[0] != len(chunks):
            raise ValueError("the number of embeddings must match the number of chunks")

        if not np.all(np.isfinite(matrix)):
            raise ValueError("embeddings must contain only finite values")

        norms = np.linalg.norm(matrix, axis=1)

        if np.any(norms == 0):
            raise ValueError("embedding vectors must not be zero")

        self._chunks = tuple(chunks)
        self._embeddings = matrix / norms[:, np.newaxis]

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return int(self._embeddings.shape[1])

    def search(
        self,
        query_embedding: npt.ArrayLike,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return chunks ranked by cosine similarity to a query vector."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query = np.asarray(query_embedding, dtype=np.float64)

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
        indices = np.argsort(-scores, kind="stable")[:result_count]

        return [
            SearchResult(
                rank=rank,
                score=float(scores[index]),
                chunk=self._chunks[int(index)],
            )
            for rank, index in enumerate(indices, start=1)
        ]
