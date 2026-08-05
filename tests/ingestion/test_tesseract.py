from pathlib import Path

import pytest

from sightcite.ingestion import TesseractOcrBackend


class FakePytesseractApi:
    def __init__(self, result: str = "Recovered text") -> None:
        self.result = result
        self.calls: list[tuple[str, str, str, float]] = []

    def image_to_string(
        self,
        image: str,
        *,
        lang: str,
        config: str,
        timeout: float,
    ) -> str:
        self.calls.append((image, lang, config, timeout))
        return self.result


def test_tesseract_extracts_text_with_configured_options(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.touch()

    api = FakePytesseractApi()
    backend = TesseractOcrBackend(
        language="fra",
        page_segmentation_mode=6,
        timeout_seconds=12.5,
        api=api,
    )

    text = backend.extract_text(image_path)

    assert text == "Recovered text"
    assert api.calls == [
        (
            str(image_path),
            "fra",
            "--psm 6",
            12.5,
        )
    ]


def test_tesseract_rejects_missing_image(tmp_path: Path) -> None:
    backend = TesseractOcrBackend(api=FakePytesseractApi())

    with pytest.raises(FileNotFoundError, match="Page image does not exist"):
        backend.extract_text(tmp_path / "missing.png")


@pytest.mark.parametrize("language", ["", "   "])
def test_tesseract_rejects_blank_language(language: str) -> None:
    with pytest.raises(ValueError, match="language must not be blank"):
        TesseractOcrBackend(
            language=language,
            api=FakePytesseractApi(),
        )


@pytest.mark.parametrize("page_segmentation_mode", [-1, 14])
def test_tesseract_rejects_invalid_page_segmentation_mode(
    page_segmentation_mode: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="page_segmentation_mode must be between 0 and 13",
    ):
        TesseractOcrBackend(
            page_segmentation_mode=page_segmentation_mode,
            api=FakePytesseractApi(),
        )


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0])
def test_tesseract_rejects_non_positive_timeout(
    timeout_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="timeout_seconds must be greater than zero",
    ):
        TesseractOcrBackend(
            timeout_seconds=timeout_seconds,
            api=FakePytesseractApi(),
        )
