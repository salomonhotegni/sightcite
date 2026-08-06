"""Text embedding interfaces and implementations."""

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from sightcite.retrieval._validation import (
    require_positive_dimension,
    validate_embedding_matrix,
)

DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class TextEmbedder(Protocol):
    """Interface implemented by text embedding models."""

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> npt.NDArray[np.float64]:
        """Embed document passages."""
        ...

    def embed_query(self, query: str) -> npt.NDArray[np.float64]:
        """Embed a retrieval query."""
        ...


class SentenceEncoderBackend(Protocol):
    """Subset of SentenceTransformer used by SightCite."""

    def get_embedding_dimension(self) -> int | None:
        """Return the model embedding dimension."""
        ...

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> object:
        """Encode text into vectors."""
        ...


class BgeTextEmbedder:
    """BGE sentence embedder for passage retrieval."""

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        backend: SentenceEncoderBackend | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if backend is None:
            loaded_model = SentenceTransformer(model_name, device=device)
            backend = cast(SentenceEncoderBackend, loaded_model)

        dimension = require_positive_dimension(backend.get_embedding_dimension())

        self._backend = backend
        self._batch_size = batch_size
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the model embedding dimension."""
        return self._dimension

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> npt.NDArray[np.float64]:
        """Embed passages without a query instruction."""
        passages = list(texts)

        if not passages:
            return np.empty((0, self.dimension), dtype=np.float64)

        return self._encode(passages)

    def embed_query(self, query: str) -> npt.NDArray[np.float64]:
        """Embed a query using the BGE retrieval instruction."""
        if not query.strip():
            raise ValueError("query must not be blank")

        matrix = self._encode([BGE_QUERY_INSTRUCTION + query])

        embedding = np.asarray(matrix[0], dtype=np.float64)

        return embedding

    def _encode(
        self,
        texts: list[str],
    ) -> npt.NDArray[np.float64]:
        encoded = self._backend.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return validate_embedding_matrix(
            encoded,
            row_count=len(texts),
            dimension=self.dimension,
        )
