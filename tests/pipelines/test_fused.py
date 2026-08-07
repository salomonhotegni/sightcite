from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sightcite.pipelines import FusedRetrievalPipeline


class FakeTextEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            [[1.0, 0.0] if "First" in text else [0.0, 1.0] for text in texts],
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


class FakeVisualEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed_images(
        self,
        image_paths: Sequence[Path],
    ) -> npt.NDArray[np.float64]:
        vectors = {
            "page_0001.png": [0.0, 1.0],
            "page_0002.png": [1.0, 0.0],
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


def test_fused_pipeline_combines_text_and_visual_rankings(
    sample_pdf: Path,
) -> None:
    with FusedRetrievalPipeline(
        sample_pdf,
        FakeTextEmbedder(),
        FakeVisualEmbedder(),
    ) as pipeline:
        results = pipeline.search(
            "first",
            top_k=2,
        )

        assert pipeline.source == sample_pdf
        assert pipeline.page_count == 2
        assert len(pipeline.pages) == 2
        assert len(pipeline.text_pages) == 2
        assert len(pipeline.chunks) == 2
        assert [result.page.page_number for result in results] == [1, 2]
        assert results[0].source_ranks == (
            ("text", 1),
            ("visual", 2),
        )
        assert all(result.page.image_path.is_file() for result in results)


def test_fused_pipeline_supports_source_weights(
    sample_pdf: Path,
) -> None:
    with FusedRetrievalPipeline(
        sample_pdf,
        FakeTextEmbedder(),
        FakeVisualEmbedder(),
        visual_weight=2.0,
    ) as pipeline:
        results = pipeline.search(
            "first",
            top_k=1,
        )

        assert results[0].page.page_number == 2


def test_fused_pipeline_releases_temporary_images(
    sample_pdf: Path,
) -> None:
    pipeline = FusedRetrievalPipeline(
        sample_pdf,
        FakeTextEmbedder(),
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


def test_fused_pipeline_rejects_negative_rank_constant(
    sample_pdf: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="rank_constant must not be negative",
    ):
        FusedRetrievalPipeline(
            sample_pdf,
            FakeTextEmbedder(),
            FakeVisualEmbedder(),
            rank_constant=-1,
        )


def test_fused_pipeline_rejects_non_positive_text_weight(
    sample_pdf: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="text_weight must be greater than zero",
    ):
        FusedRetrievalPipeline(
            sample_pdf,
            FakeTextEmbedder(),
            FakeVisualEmbedder(),
            text_weight=0.0,
        )


def test_fused_pipeline_rejects_non_positive_visual_weight(
    sample_pdf: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="visual_weight must be greater than zero",
    ):
        FusedRetrievalPipeline(
            sample_pdf,
            FakeTextEmbedder(),
            FakeVisualEmbedder(),
            visual_weight=0.0,
        )
