"""PDF page rendering utilities."""

from pathlib import Path

import pymupdf

from sightcite.ingestion.models import RenderedPage


def render_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 144,
) -> list[RenderedPage]:
    """Render every page of a PDF as a PNG image.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory in which page images will be written.
        dpi: Rendering resolution in dots per inch.

    Returns:
        Metadata for the rendered pages, ordered by page number.

    Raises:
        FileNotFoundError: If the source PDF does not exist.
        ValueError: If ``dpi`` is not positive.
    """
    source = Path(pdf_path)
    destination = Path(output_dir)

    if not source.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {source}")

    if dpi <= 0:
        raise ValueError("dpi must be greater than zero")

    rendered_pages: list[RenderedPage] = []

    with pymupdf.open(source) as document:  # type: ignore[no-untyped-call]
        destination.mkdir(parents=True, exist_ok=True)

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            image_path = destination / f"page_{page_number:04d}.png"

            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            pixmap.save(image_path)

            rendered_pages.append(
                RenderedPage(
                    page_number=page_number,
                    image_path=image_path,
                    width=pixmap.width,
                    height=pixmap.height,
                )
            )

    return rendered_pages
