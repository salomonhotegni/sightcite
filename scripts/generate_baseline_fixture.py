"""Generate the synthetic SightCite baseline benchmark PDF."""

from pathlib import Path

import pymupdf

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
OUTPUT_PATH = Path("examples/baseline/scientific-paper.pdf")


def _insert_native_page(
    document: pymupdf.Document,
    *,
    heading: str,
    body: str,
) -> None:
    page = document.new_page(
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
    )
    page.insert_text(
        (72, 90),
        heading,
        fontsize=22,
    )
    page.insert_textbox(
        pymupdf.Rect(72, 130, 540, 650),
        body,
        fontsize=14,
        lineheight=1.4,
    )


def _make_scanned_page_image(
    *,
    heading: str,
    body: str,
) -> bytes:
    with pymupdf.open() as temporary_document:  # type: ignore[no-untyped-call]
        page = temporary_document.new_page(
            width=PAGE_WIDTH,
            height=PAGE_HEIGHT,
        )
        page.insert_text(
            (72, 90),
            heading,
            fontsize=22,
        )
        page.insert_textbox(
            pymupdf.Rect(72, 130, 540, 650),
            body,
            fontsize=14,
            lineheight=1.4,
        )

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False,
        )
        return pixmap.tobytes("png")


def generate_fixture(output_path: Path = OUTPUT_PATH) -> Path:
    """Generate a PDF containing native-text and image-only pages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scanned_page = _make_scanned_page_image(
        heading="2. Microscopy Results",
        body=(
            "Electron microscopy revealed a graphene lattice spacing of "
            "0.34 nanometers. The defect density decreased after thermal "
            "annealing at 700 degrees Celsius."
        ),
    )

    with pymupdf.open() as document:  # type: ignore[no-untyped-call]
        _insert_native_page(
            document,
            heading="1. Photovoltaic Method",
            body=(
                "The perovskite solar cells reached a power conversion "
                "efficiency of 24.1 percent under standard illumination. "
                "A titanium dioxide transport layer was used."
            ),
        )

        scanned = document.new_page(
            width=PAGE_WIDTH,
            height=PAGE_HEIGHT,
        )
        scanned.insert_image(
            scanned.rect,
            stream=scanned_page,
        )

        _insert_native_page(
            document,
            heading="3. Battery Evaluation",
            body=(
                "The lithium sulfur battery retained 82 percent of its "
                "initial capacity after 500 charge cycles. The electrolyte "
                "contained a nitrate additive."
            ),
        )

        document.save(output_path)

    return output_path


def main() -> None:
    """Generate the baseline fixture and print its location."""
    output_path = generate_fixture()
    print(f"Generated benchmark PDF: {output_path}")


if __name__ == "__main__":
    main()
