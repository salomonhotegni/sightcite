from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sightcite.retrieval import ClipVisualEmbedder


class FakeClipBackend:
    def __init__(
        self,
        *,
        dimension: int | None = 2,
        output: object | None = None,
    ) -> None:
        self.dimension = dimension
        self.output = output
        self.calls: list[list[object]] = []

    def get_embedding_dimension(self) -> int | None:
        return self.dimension

    def encode(
        self,
        inputs: list[object],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> object:
        self.calls.append(inputs)

        if self.output is not None:
            return self.output

        return np.ones((len(inputs), 2))


def make_image(
    path: Path,
    *,
    color: str = "white",
) -> Path:
    with Image.new("RGB", (32, 24), color=color) as image:
        image.save(path)

    return path


def test_clip_embeds_images(tmp_path: Path) -> None:
    first = make_image(tmp_path / "first.png", color="red")
    second = make_image(tmp_path / "second.png", color="blue")
    backend = FakeClipBackend()
    embedder = ClipVisualEmbedder(
        backend=backend,
        batch_size=4,
    )

    embeddings = embedder.embed_images([first, second])

    assert embeddings.shape == (2, 2)
    assert embedder.dimension == 2
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 2
    assert all(isinstance(value, Image.Image) for value in backend.calls[0])


def test_clip_embeds_text_query() -> None:
    backend = FakeClipBackend()
    embedder = ClipVisualEmbedder(backend=backend)

    embedding = embedder.embed_query("a scientific chart showing increasing accuracy")

    assert embedding.shape == (2,)
    assert backend.calls == [["a scientific chart showing increasing accuracy"]]


def test_clip_handles_empty_image_input() -> None:
    backend = FakeClipBackend()
    embedder = ClipVisualEmbedder(backend=backend)

    embeddings = embedder.embed_images([])

    assert embeddings.shape == (0, 2)
    assert backend.calls == []


def test_clip_rejects_missing_image(tmp_path: Path) -> None:
    embedder = ClipVisualEmbedder(backend=FakeClipBackend())

    with pytest.raises(
        FileNotFoundError,
        match="Page image does not exist",
    ):
        embedder.embed_images([tmp_path / "missing.png"])


@pytest.mark.parametrize("batch_size", [0, -1])
def test_clip_rejects_invalid_batch_size(
    batch_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        ClipVisualEmbedder(
            backend=FakeClipBackend(),
            batch_size=batch_size,
        )


@pytest.mark.parametrize("dimension", [None, 0, -1])
def test_clip_rejects_invalid_dimension(
    dimension: int | None,
) -> None:
    with pytest.raises(
        ValueError,
        match="must report a positive dimension",
    ):
        ClipVisualEmbedder(backend=FakeClipBackend(dimension=dimension))


def test_clip_rejects_blank_query() -> None:
    embedder = ClipVisualEmbedder(backend=FakeClipBackend())

    with pytest.raises(
        ValueError,
        match="query must not be blank",
    ):
        embedder.embed_query("   ")


def test_clip_rejects_unexpected_output_shape() -> None:
    backend = FakeClipBackend(
        output=np.ones((1, 3)),
    )
    embedder = ClipVisualEmbedder(backend=backend)

    with pytest.raises(ValueError, match="expected"):
        embedder.embed_query("chart")


def test_clip_rejects_non_finite_output(
    tmp_path: Path,
) -> None:
    image_path = make_image(tmp_path / "page.png")
    backend = FakeClipBackend(
        output=np.array([[np.inf, 0.0]]),
    )
    embedder = ClipVisualEmbedder(backend=backend)

    with pytest.raises(ValueError, match="non-finite"):
        embedder.embed_images([image_path])
