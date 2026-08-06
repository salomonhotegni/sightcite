"""Visual embedding interfaces."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt


class VisualEmbedder(Protocol):
    """Embed page images and text queries into one shared vector space."""

    @property
    def dimension(self) -> int:
        """Return the shared embedding dimension."""
        ...

    def embed_images(
        self,
        image_paths: Sequence[Path],
    ) -> npt.NDArray[np.float64]:
        """Embed rendered page images."""
        ...

    def embed_query(
        self,
        query: str,
    ) -> npt.NDArray[np.float64]:
        """Embed a textual retrieval query."""
        ...
