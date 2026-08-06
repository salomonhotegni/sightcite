"""CLIP visual embedding implementation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from PIL import Image
from sentence_transformers import SentenceTransformer

from sightcite.retrieval._validation import (
    require_positive_dimension,
    validate_embedding_matrix,
)

DEFAULT_CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"


class ClipEncoderBackend(Protocol):
    """Subset of SentenceTransformer used for CLIP encoding."""

    def get_embedding_dimension(self) -> int | None:
        """Return the shared image-text embedding dimension."""
        ...

    def encode(
        self,
        inputs: list[object],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> object:
        """Encode images or text into shared vectors."""
        ...


class ClipVisualEmbedder:
    """Embed page images and text queries using CLIP."""

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 16,
        backend: ClipEncoderBackend | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if backend is None:
            loaded_model = SentenceTransformer(
                model_name,
                device=device,
            )
            backend = cast(ClipEncoderBackend, loaded_model)

        self._dimension = require_positive_dimension(backend.get_embedding_dimension())
        self._backend = backend
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        """Return the shared image-text embedding dimension."""
        return self._dimension

    def embed_images(
        self,
        image_paths: Sequence[Path],
    ) -> npt.NDArray[np.float64]:
        """Embed rendered page images."""
        paths = list(image_paths)

        if not paths:
            return np.empty(
                (0, self.dimension),
                dtype=np.float64,
            )

        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Page image does not exist: {path}")

        images: list[Image.Image] = []

        try:
            for path in paths:
                with Image.open(path) as opened_image:
                    images.append(opened_image.convert("RGB"))

            return self._encode(
                cast(list[object], images),
                row_count=len(images),
            )
        finally:
            for converted_image in images:
                converted_image.close()

    def embed_query(
        self,
        query: str,
    ) -> npt.NDArray[np.float64]:
        """Embed a textual retrieval query."""
        if not query.strip():
            raise ValueError("query must not be blank")

        matrix = self._encode(
            [query],
            row_count=1,
        )

        return np.asarray(matrix[0], dtype=np.float64)

    def _encode(
        self,
        inputs: list[object],
        *,
        row_count: int,
    ) -> npt.NDArray[np.float64]:
        encoded = self._backend.encode(
            inputs,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return validate_embedding_matrix(
            encoded,
            row_count=row_count,
            dimension=self.dimension,
        )
