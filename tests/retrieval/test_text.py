from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from sightcite.ingestion import TextChunk
from sightcite.retrieval import TextRetriever


class FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> npt.NDArray[np.float64]:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float64)

        vectors = {
            "cats": [1.0, 0.0],
            "dogs": [0.0, 1.0],
            "cats and dogs": [0.8, 0.6],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float64)

    def embed_query(self, query: str) -> npt.NDArray[np.float64]:
        vectors = {
            "cats": [1.0, 0.0],
            "dogs": [0.0, 1.0],
        }
        return np.asarray(vectors[query], dtype=np.float64)


def test_text_retriever_embeds_and_ranks_chunks() -> None:
    chunks = [
        TextChunk(1, 0, "cats", 0, 1),
        TextChunk(2, 0, "dogs", 0, 1),
        TextChunk(3, 0, "cats and dogs", 0, 3),
    ]
    retriever = TextRetriever(chunks, FakeEmbedder())

    results = retriever.search("cats", top_k=2)

    assert retriever.size == 3
    assert [result.chunk.page_number for result in results] == [1, 3]
    assert [result.rank for result in results] == [1, 2]


def test_text_retriever_supports_empty_index() -> None:
    retriever = TextRetriever([], FakeEmbedder())

    assert retriever.size == 0
    assert retriever.search("cats") == []
