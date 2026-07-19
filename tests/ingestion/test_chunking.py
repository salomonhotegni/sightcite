import pytest

from sightcite.ingestion import ExtractedPage, chunk_extracted_pages


def test_chunk_extracted_pages_adds_overlap() -> None:
    pages = [
        ExtractedPage(
            page_number=1,
            text="zero one two three four five six seven eight nine",
        )
    ]

    chunks = chunk_extracted_pages(pages, chunk_size=4, overlap=1)

    assert [chunk.text for chunk in chunks] == [
        "zero one two three",
        "three four five six",
        "six seven eight nine",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [(chunk.start_word, chunk.end_word) for chunk in chunks] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]


def test_chunk_extracted_pages_preserves_page_boundaries() -> None:
    pages = [
        ExtractedPage(page_number=1, text="alpha beta gamma"),
        ExtractedPage(page_number=2, text=""),
        ExtractedPage(page_number=3, text="delta epsilon"),
    ]

    chunks = chunk_extracted_pages(pages, chunk_size=2, overlap=0)

    assert [chunk.page_number for chunk in chunks] == [1, 1, 3]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 0]
    assert [chunk.text for chunk in chunks] == [
        "alpha beta",
        "gamma",
        "delta epsilon",
    ]


def test_chunk_extracted_pages_accepts_empty_input() -> None:
    assert chunk_extracted_pages([]) == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_extracted_pages_rejects_invalid_chunk_size(
    chunk_size: int,
) -> None:
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        chunk_extracted_pages([], chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [-1, -10])
def test_chunk_extracted_pages_rejects_negative_overlap(
    overlap: int,
) -> None:
    with pytest.raises(ValueError, match="overlap must not be negative"):
        chunk_extracted_pages([], overlap=overlap)


@pytest.mark.parametrize("overlap", [200, 201])
def test_chunk_extracted_pages_rejects_large_overlap(
    overlap: int,
) -> None:
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_extracted_pages([], chunk_size=200, overlap=overlap)
