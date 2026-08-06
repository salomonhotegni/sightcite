import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sightcite.retrieval import ClipVisualEmbedder

pytestmark = pytest.mark.model


@pytest.mark.skipif(
    os.environ.get("SIGHTCITE_RUN_MODEL_TESTS") != "1",
    reason="Set SIGHTCITE_RUN_MODEL_TESTS=1 to run model tests",
)
def test_clip_embeds_real_image_and_query(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scientific-figure.png"

    with Image.new(
        "RGB",
        (224, 224),
        color="red",
    ) as image:
        image.save(image_path)

    embedder = ClipVisualEmbedder(
        device="cpu",
        batch_size=1,
    )

    image_embeddings = embedder.embed_images([image_path])
    query_embedding = embedder.embed_query("a red scientific figure")

    assert embedder.dimension > 0
    assert image_embeddings.shape == (1, embedder.dimension)
    assert query_embedding.shape == (embedder.dimension,)
    assert np.all(np.isfinite(image_embeddings))
    assert np.all(np.isfinite(query_embedding))
    assert np.linalg.norm(image_embeddings[0]) == pytest.approx(
        1.0,
        abs=1e-5,
    )
    assert np.linalg.norm(query_embedding) == pytest.approx(
        1.0,
        abs=1e-5,
    )
