from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sightcite.pipelines import VisualRetrievalPipeline


class FakeVisualEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed_images(
        self,
        image_paths: Sequence[Path],
    ) -> npt.NDArray[np.float64]:
        vectors = {
            "page_0001.png": [1.0, 0.0],
            "page_0002.png": [0.0, 1.0],
        }
        return np.asarray(
            [vectors[path.name] for path in image_paths],
            dtype=np.float64,
        )

    def embed_query(
        self,
        query: str,
    ) -> npt.NDArray[np.float64]:
        if query == "first":
            return np.asarray(
                [1.0, 0.0],
                dtype=np.float64,
            )

        return np.asarray(
            [0.0, 1.0],
            dtype=np.float64,
        )


def test_visual_pipeline_renders_indexes_and_searches_pdf(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "rendered"

    pipeline = VisualRetrievalPipeline(
        sample_pdf,
        FakeVisualEmbedder(),
        output_dir=output_dir,
        dpi=100,
    )

    results = pipeline.search(
        "second",
        top_k=1,
    )

    assert pipeline.source == sample_pdf
    assert pipeline.page_count == 2
    assert len(pipeline.pages) == 2
    assert all(page.image_path.is_file() for page in pipeline.pages)
    assert results[0].page.page_number == 2
    assert results[0].page.image_path.parent == output_dir

    pipeline.close()

    assert all(page.image_path.is_file() for page in pipeline.pages)


def test_visual_pipeline_manages_temporary_images(
    sample_pdf: Path,
) -> None:
    pipeline = VisualRetrievalPipeline(
        sample_pdf,
        FakeVisualEmbedder(),
    )
    image_paths = [page.image_path for page in pipeline.pages]

    assert all(path.is_file() for path in image_paths)

    pipeline.close()

    assert all(not path.exists() for path in image_paths)

    with pytest.raises(
        RuntimeError,
        match="visual retrieval pipeline is closed",
    ):
        pipeline.search("first")


def test_visual_pipeline_supports_context_manager(
    sample_pdf: Path,
) -> None:
    with VisualRetrievalPipeline(
        sample_pdf,
        FakeVisualEmbedder(),
    ) as pipeline:
        image_paths = [page.image_path for page in pipeline.pages]
        results = pipeline.search("first", top_k=1)

        assert results[0].page.page_number == 1
        assert all(path.is_file() for path in image_paths)

    assert all(not path.exists() for path in image_paths)


def test_visual_pipeline_accepts_string_path(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    pipeline = VisualRetrievalPipeline(
        str(sample_pdf),
        FakeVisualEmbedder(),
        output_dir=tmp_path / "rendered",
    )

    assert pipeline.source == sample_pdf
    pipeline.close()
