import pytest

from sightcite.indexing import (
    INDEX_SCHEMA_VERSION,
    DocumentIndexManifest,
    TextIndexConfiguration,
    VisualIndexConfiguration,
)


def make_text_configuration() -> TextIndexConfiguration:
    return TextIndexConfiguration(
        model_name="BAAI/bge-small-en-v1.5",
        dimension=384,
        chunk_size=200,
        overlap=40,
    )


def make_visual_configuration() -> VisualIndexConfiguration:
    return VisualIndexConfiguration(
        model_name="openai/clip-vit-base-patch32",
        dimension=512,
        dpi=144,
    )


def make_manifest(
    *,
    source_sha256: str = "a" * 64,
    schema_version: int = INDEX_SCHEMA_VERSION,
) -> DocumentIndexManifest:
    return DocumentIndexManifest(
        document_id="paper-001",
        source_filename="paper.pdf",
        source_sha256=source_sha256,
        page_count=12,
        text=make_text_configuration(),
        visual=make_visual_configuration(),
        schema_version=schema_version,
    )


def test_index_configurations_accept_valid_values() -> None:
    text = TextIndexConfiguration(
        model_name="text-model",
        dimension=384,
        chunk_size=200,
        overlap=40,
        ocr_enabled=True,
        min_native_chars=20,
        ocr_dpi=180,
    )
    visual = VisualIndexConfiguration(
        model_name="visual-model",
        dimension=512,
        dpi=180,
    )

    assert text.ocr_enabled is True
    assert text.ocr_dpi == 180
    assert visual.dimension == 512
    assert visual.dpi == 180


@pytest.mark.parametrize("model_name", ["", "   "])
def test_text_configuration_rejects_blank_model_name(
    model_name: str,
) -> None:
    with pytest.raises(ValueError, match="text model name must not be blank"):
        TextIndexConfiguration(
            model_name=model_name,
            dimension=384,
            chunk_size=200,
            overlap=40,
        )


@pytest.mark.parametrize("dimension", [0, -1])
def test_text_configuration_rejects_non_positive_dimension(
    dimension: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="text embedding dimension must be greater than zero",
    ):
        TextIndexConfiguration(
            model_name="text-model",
            dimension=dimension,
            chunk_size=200,
            overlap=40,
        )


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_text_configuration_rejects_non_positive_chunk_size(
    chunk_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="chunk size must be greater than zero",
    ):
        TextIndexConfiguration(
            model_name="text-model",
            dimension=384,
            chunk_size=chunk_size,
            overlap=0,
        )


@pytest.mark.parametrize(
    ("overlap", "message"),
    [
        (-1, "overlap must not be negative"),
        (200, "overlap must be smaller than chunk size"),
        (201, "overlap must be smaller than chunk size"),
    ],
)
def test_text_configuration_rejects_invalid_overlap(
    overlap: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TextIndexConfiguration(
            model_name="text-model",
            dimension=384,
            chunk_size=200,
            overlap=overlap,
        )


def test_text_configuration_rejects_negative_minimum_native_chars() -> None:
    with pytest.raises(
        ValueError,
        match="minimum native characters must not be negative",
    ):
        TextIndexConfiguration(
            model_name="text-model",
            dimension=384,
            chunk_size=200,
            overlap=40,
            min_native_chars=-1,
        )


@pytest.mark.parametrize("dpi", [0, -1])
def test_text_configuration_rejects_non_positive_ocr_dpi(
    dpi: int,
) -> None:
    with pytest.raises(ValueError, match="OCR dpi must be greater than zero"):
        TextIndexConfiguration(
            model_name="text-model",
            dimension=384,
            chunk_size=200,
            overlap=40,
            ocr_dpi=dpi,
        )


@pytest.mark.parametrize("model_name", ["", "   "])
def test_visual_configuration_rejects_blank_model_name(
    model_name: str,
) -> None:
    with pytest.raises(ValueError, match="visual model name must not be blank"):
        VisualIndexConfiguration(
            model_name=model_name,
            dimension=512,
        )


@pytest.mark.parametrize("dimension", [0, -1])
def test_visual_configuration_rejects_non_positive_dimension(
    dimension: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="visual embedding dimension must be greater than zero",
    ):
        VisualIndexConfiguration(
            model_name="visual-model",
            dimension=dimension,
        )


@pytest.mark.parametrize("dpi", [0, -1])
def test_visual_configuration_rejects_non_positive_dpi(
    dpi: int,
) -> None:
    with pytest.raises(ValueError, match="visual dpi must be greater than zero"):
        VisualIndexConfiguration(
            model_name="visual-model",
            dimension=512,
            dpi=dpi,
        )


def test_document_manifest_accepts_valid_metadata() -> None:
    manifest = make_manifest()

    assert manifest.schema_version == INDEX_SCHEMA_VERSION
    assert manifest.document_id == "paper-001"
    assert manifest.source_filename == "paper.pdf"
    assert manifest.source_sha256 == "a" * 64
    assert manifest.page_count == 12


@pytest.mark.parametrize("document_id", ["", "   "])
def test_document_manifest_rejects_blank_document_id(
    document_id: str,
) -> None:
    with pytest.raises(ValueError, match="document id must not be blank"):
        DocumentIndexManifest(
            document_id=document_id,
            source_filename="paper.pdf",
            source_sha256="a" * 64,
            page_count=1,
            text=make_text_configuration(),
            visual=make_visual_configuration(),
        )


@pytest.mark.parametrize(
    "source_filename",
    [
        "",
        "   ",
        "papers/paper.pdf",
        "../paper.pdf",
    ],
)
def test_document_manifest_rejects_invalid_source_filename(
    source_filename: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="source filename",
    ):
        DocumentIndexManifest(
            document_id="paper-001",
            source_filename=source_filename,
            source_sha256="a" * 64,
            page_count=1,
            text=make_text_configuration(),
            visual=make_visual_configuration(),
        )


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
    ],
)
def test_document_manifest_rejects_invalid_sha256(
    digest: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="source sha256 must be a lowercase 64-character hexadecimal digest",
    ):
        make_manifest(source_sha256=digest)


@pytest.mark.parametrize("page_count", [0, -1])
def test_document_manifest_rejects_non_positive_page_count(
    page_count: int,
) -> None:
    with pytest.raises(ValueError, match="page count must be greater than zero"):
        DocumentIndexManifest(
            document_id="paper-001",
            source_filename="paper.pdf",
            source_sha256="a" * 64,
            page_count=page_count,
            text=make_text_configuration(),
            visual=make_visual_configuration(),
        )


def test_document_manifest_rejects_unknown_schema_version() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported index schema version: 2",
    ):
        make_manifest(schema_version=2)
