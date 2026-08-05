"""Tesseract implementation of the OCR backend protocol."""

from pathlib import Path
from typing import Protocol, cast

import pytesseract  # type: ignore[import-untyped]


class _PytesseractApi(Protocol):
    """Typed subset of pytesseract used by SightCite."""

    def image_to_string(
        self,
        image: str,
        *,
        lang: str,
        config: str,
        timeout: float,
    ) -> str:
        """Extract text from an image."""
        ...


class TesseractOcrBackend:
    """Extract page text using a local Tesseract installation."""

    def __init__(
        self,
        *,
        language: str = "eng",
        page_segmentation_mode: int = 3,
        timeout_seconds: float = 30.0,
        api: _PytesseractApi | None = None,
    ) -> None:
        if not language.strip():
            raise ValueError("language must not be blank")

        if not 0 <= page_segmentation_mode <= 13:
            raise ValueError("page_segmentation_mode must be between 0 and 13")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self._language = language
        self._page_segmentation_mode = page_segmentation_mode
        self._timeout_seconds = timeout_seconds
        self._api = api or cast(_PytesseractApi, pytesseract)

    def extract_text(self, image_path: Path) -> str:
        """Extract text from one rendered page image."""
        if not image_path.is_file():
            raise FileNotFoundError(f"Page image does not exist: {image_path}")

        return self._api.image_to_string(
            str(image_path),
            lang=self._language,
            config=f"--psm {self._page_segmentation_mode}",
            timeout=self._timeout_seconds,
        )
