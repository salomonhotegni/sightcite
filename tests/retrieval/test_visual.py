from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sightcite.ingestion import RenderedPage
from sightcite.retrieval import VisualRetriever


class FakeVisualEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed_images(
        self,
        image_paths: Sequence[Path],
    ) -> npt.NDArray[np.float64]:
        if not image_paths:
            return np.empty(
                (0, self.dimension),
                dtype=np.float64,
            )

        vectors = {
            "page-1.png": [1.0, 0.0],
            "page-2.png": [0.0, 1.0],
            "page-3.png": [0.8, 0.6],
        }

        return np.asarray(
            [vectors[path.name] for path in image_paths],
            dtype=np.float64,
        )

    def embed_query(
        self,
        query: str,
    ) -> npt.NDArray[np.float64]:
        vectors = {
            "diagram": [1.0, 0.0],
            "table": [0.0, 1.0],
        }
        return np.asarray(
            vectors[query],
            dtype=np.float64,
        )


def make_pages() -> list[RenderedPage]:
    return [
        RenderedPage(
            page_number=1,
            image_path=Path("page-1.png"),
            width=600,
            height=800,
        ),
        RenderedPage(
            page_number=2,
            image_path=Path("page-2.png"),
            width=600,
            height=800,
        ),
        RenderedPage(
            page_number=3,
            image_path=Path("page-3.png"),
            width=600,
            height=800,
        ),
    ]


def test_visual_retriever_embeds_and_ranks_pages() -> None:
    retriever = VisualRetriever(
        make_pages(),
        FakeVisualEmbedder(),
    )

    results = retriever.search(
        "diagram",
        top_k=2,
    )

    assert retriever.size == 3
    assert [result.page.page_number for result in results] == [1, 3]
    assert [result.rank for result in results] == [1, 2]


def test_visual_retriever_supports_empty_index() -> None:
    retriever = VisualRetriever(
        [],
        FakeVisualEmbedder(),
    )

    assert retriever.size == 0
    assert retriever.search("diagram") == []
