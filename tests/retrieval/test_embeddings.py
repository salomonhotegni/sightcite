import numpy as np
import pytest

from sightcite.retrieval import BGE_QUERY_INSTRUCTION, BgeTextEmbedder


class FakeBackend:
    def __init__(
        self,
        *,
        dimension: int | None = 2,
        output: object | None = None,
    ) -> None:
        self.dimension = dimension
        self.output = output
        self.calls: list[list[str]] = []

    def get_embedding_dimension(self) -> int | None:
        return self.dimension

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> object:
        self.calls.append(sentences)

        if self.output is not None:
            return self.output

        return np.ones((len(sentences), 2))


def test_embed_documents_does_not_add_query_instruction() -> None:
    backend = FakeBackend()
    embedder = BgeTextEmbedder(backend=backend, batch_size=8)

    embeddings = embedder.embed_documents(["first passage", "second passage"])

    assert embeddings.shape == (2, 2)
    assert backend.calls == [["first passage", "second passage"]]
    assert embedder.dimension == 2


def test_embed_query_adds_bge_instruction() -> None:
    backend = FakeBackend()
    embedder = BgeTextEmbedder(backend=backend)

    embedding = embedder.embed_query("What is the best method?")

    assert embedding.shape == (2,)
    assert backend.calls == [[BGE_QUERY_INSTRUCTION + "What is the best method?"]]


def test_embed_documents_handles_empty_input_without_backend_call() -> None:
    backend = FakeBackend()
    embedder = BgeTextEmbedder(backend=backend)

    embeddings = embedder.embed_documents([])

    assert embeddings.shape == (0, 2)
    assert backend.calls == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_embedder_rejects_invalid_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        BgeTextEmbedder(backend=FakeBackend(), batch_size=batch_size)


@pytest.mark.parametrize("dimension", [None, 0, -1])
def test_embedder_rejects_invalid_dimension(dimension: int | None) -> None:
    with pytest.raises(ValueError, match="must report a positive dimension"):
        BgeTextEmbedder(backend=FakeBackend(dimension=dimension))


def test_embedder_rejects_blank_query() -> None:
    embedder = BgeTextEmbedder(backend=FakeBackend())

    with pytest.raises(ValueError, match="query must not be blank"):
        embedder.embed_query("   ")


def test_embedder_rejects_unexpected_output_shape() -> None:
    backend = FakeBackend(output=np.ones((1, 3)))
    embedder = BgeTextEmbedder(backend=backend)

    with pytest.raises(ValueError, match="expected"):
        embedder.embed_documents(["passage"])


def test_embedder_rejects_non_finite_output() -> None:
    backend = FakeBackend(output=np.array([[np.nan, 1.0]]))
    embedder = BgeTextEmbedder(backend=backend)

    with pytest.raises(ValueError, match="non-finite"):
        embedder.embed_documents(["passage"])
