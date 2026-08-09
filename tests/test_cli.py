import json
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt
import pytest

from sightcite import cli
from sightcite.answering import EvidencePage, GroundedAnswer
from sightcite.integrations.langchain import AnswerRequest


class FakeBgeTextEmbedder:
    """Deterministic replacement for the CLI's real model."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None,
        batch_size: int,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> npt.NDArray[np.float64]:
        vectors = [[1.0, 0.0] if "First" in text else [0.0, 1.0] for text in texts]
        return np.asarray(vectors, dtype=np.float64)

    def embed_query(self, query: str) -> npt.NDArray[np.float64]:
        if "first" in query.lower():
            return np.asarray([1.0, 0.0], dtype=np.float64)

        return np.asarray([0.0, 1.0], dtype=np.float64)


class FakeClipVisualEmbedder:
    """Deterministic replacement for the CLI CLIP model."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None,
        batch_size: int,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

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
        if "first" in query.lower():
            return np.asarray(
                [1.0, 0.0],
                dtype=np.float64,
            )

        return np.asarray(
            [0.0, 1.0],
            dtype=np.float64,
        )


def write_benchmark_dataset(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "document_id": "sample-paper",
        "pdf_path": "sample.pdf",
        "questions": [
            {
                "query_id": "q1",
                "query": "What is on the first page?",
                "relevant_pages": [1],
            },
            {
                "query_id": "q2",
                "query": "What is on the second page?",
                "relevant_pages": [2],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_benchmark_command_writes_report(
    sample_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "benchmark.json"
    report_path = tmp_path / "report.json"
    write_benchmark_dataset(dataset_path)
    monkeypatch.setattr(cli, "BgeTextEmbedder", FakeBgeTextEmbedder)

    exit_code = cli.main(
        [
            "benchmark",
            str(dataset_path),
            "--output",
            str(report_path),
            "--device",
            "cpu",
            "--chunk-size",
            "20",
            "--overlap",
            "5",
            "--batch-size",
            "4",
        ]
    )

    output = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "Document: sample-paper" in output
    assert "Recall@1: 1.0000" in output
    assert "MRR: 1.0000" in output
    assert f"Report: {report_path}" in output
    assert report["metrics"]["recall_at_1"] == 1.0


def test_benchmark_command_rejects_large_overlap(
    sample_pdf: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "benchmark.json"
    write_benchmark_dataset(dataset_path)

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "benchmark",
                str(dataset_path),
                "--output",
                str(tmp_path / "report.json"),
                "--chunk-size",
                "10",
                "--overlap",
                "10",
            ]
        )

    assert error.value.code == 2
    assert "overlap must be smaller than chunk-size" in capsys.readouterr().err


def test_benchmark_command_rejects_missing_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "benchmark",
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )

    assert error.value.code == 2
    assert "benchmark dataset does not exist" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--chunk-size", "0", "value must be greater than zero"),
        ("--batch-size", "-1", "value must be greater than zero"),
        ("--overlap", "-1", "value must not be negative"),
        ("--ocr-min-native-chars", "-1", "value must not be negative"),
        ("--ocr-dpi", "0", "value must be greater than zero"),
        (
            "--ocr-page-segmentation-mode",
            "14",
            "value must be between 0 and 13",
        ),
        ("--ocr-timeout", "0", "value must be greater than zero"),
        ("--visual-dpi", "0", "value must be greater than zero"),
    ],
)
def test_benchmark_command_rejects_invalid_numeric_option(
    option: str,
    value: str,
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "benchmark",
                "benchmark.json",
                "--output",
                str(tmp_path / "report.json"),
                option,
                value,
            ]
        )

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_benchmark_command_supports_ocr(
    blank_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "ocr-benchmark.json"
    report_path = tmp_path / "ocr-report.json"
    ocr_output_dir = tmp_path / "ocr-pages"

    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "document_id": "scanned-paper",
                "pdf_path": blank_pdf.name,
                "questions": [
                    {
                        "query_id": "q1",
                        "query": "What is on the first page?",
                        "relevant_pages": [1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeTesseractOcrBackend:
        def __init__(
            self,
            *,
            language: str,
            page_segmentation_mode: int,
            timeout_seconds: float,
        ) -> None:
            captured["configuration"] = (
                language,
                page_segmentation_mode,
                timeout_seconds,
            )

        def extract_text(self, image_path: Path) -> str:
            captured["image_path"] = image_path
            return "First page recovered with OCR"

    monkeypatch.setattr(cli, "BgeTextEmbedder", FakeBgeTextEmbedder)
    monkeypatch.setattr(
        cli,
        "TesseractOcrBackend",
        FakeTesseractOcrBackend,
    )

    exit_code = cli.main(
        [
            "benchmark",
            str(dataset_path),
            "--output",
            str(report_path),
            "--ocr",
            "--ocr-language",
            "fra",
            "--ocr-page-segmentation-mode",
            "6",
            "--ocr-timeout",
            "12.5",
            "--ocr-min-native-chars",
            "25",
            "--ocr-dpi",
            "200",
            "--ocr-output-dir",
            str(ocr_output_dir),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    image_path = captured["image_path"]

    assert exit_code == 0
    assert captured["configuration"] == ("fra", 6, 12.5)
    assert isinstance(image_path, Path)
    assert image_path.parent == ocr_output_dir
    assert image_path.is_file()
    assert report["system_name"] == "bge-text-ocr"
    assert report["metrics"]["recall_at_1"] == 1.0


def test_benchmark_command_supports_visual_retrieval(
    sample_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "benchmark.json"
    report_path = tmp_path / "visual-report.json"
    output_dir = tmp_path / "visual-pages"
    write_benchmark_dataset(dataset_path)

    monkeypatch.setattr(
        cli,
        "ClipVisualEmbedder",
        FakeClipVisualEmbedder,
    )

    exit_code = cli.main(
        [
            "benchmark",
            str(dataset_path),
            "--output",
            str(report_path),
            "--visual",
            "--model",
            "test-clip",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--visual-dpi",
            "100",
            "--visual-output-dir",
            str(output_dir),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["system_name"] == "clip-visual"
    assert report["metrics"]["recall_at_1"] == 1.0
    assert (output_dir / "page_0001.png").is_file()
    assert (output_dir / "page_0002.png").is_file()


def test_benchmark_command_rejects_ocr_and_visual_together(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "benchmark",
                "benchmark.json",
                "--output",
                str(tmp_path / "report.json"),
                "--ocr",
                "--visual",
            ]
        )

    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_answer_command_runs_grounded_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()

    image_path = tmp_path / "page-2.png"
    image_path.touch()

    expected_answer = GroundedAnswer(
        question="What did the experiment show?",
        text="The experiment showed improved performance.",
        citations=(
            EvidencePage(
                page_number=2,
                image_path=image_path,
                retrieval_rank=1,
                retrieval_score=0.75,
                source_ranks=(
                    ("text", 1),
                    ("visual", 2),
                ),
            ),
        ),
    )
    captured: dict[str, object] = {}

    class FakeOpenAIVisionLanguageModel:
        def __init__(
            self,
            model: str,
            *,
            image_detail: str,
        ) -> None:
            captured["answer_model"] = model
            captured["image_detail"] = image_detail

    class FakeGroundedAnswerService:
        def __init__(
            self,
            model: object,
        ) -> None:
            captured["answer_service_model"] = model

    class FakeFusedRetrievalPipeline:
        def __init__(
            self,
            pdf_path: str | Path,
            text_embedder: object,
            visual_embedder: object,
            **configuration: object,
        ) -> None:
            captured["pdf_path"] = Path(pdf_path)
            captured["text_embedder"] = text_embedder
            captured["visual_embedder"] = visual_embedder
            captured["pipeline_configuration"] = configuration

        def __enter__(self) -> "FakeFusedRetrievalPipeline":
            captured["pipeline_entered"] = True
            return self

        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            captured["pipeline_exited"] = True

    class FakeChain:
        def invoke(
            self,
            request: AnswerRequest,
        ) -> GroundedAnswer:
            captured["request"] = request
            return expected_answer

    def fake_create_grounded_answer_chain(
        retrieval_pipeline: object,
        answer_service: object,
    ) -> FakeChain:
        captured["chain_retrieval_pipeline"] = retrieval_pipeline
        captured["chain_answer_service"] = answer_service
        return FakeChain()

    monkeypatch.setattr(cli, "BgeTextEmbedder", FakeBgeTextEmbedder)
    monkeypatch.setattr(cli, "ClipVisualEmbedder", FakeClipVisualEmbedder)
    monkeypatch.setattr(
        cli,
        "OpenAIVisionLanguageModel",
        FakeOpenAIVisionLanguageModel,
    )
    monkeypatch.setattr(
        cli,
        "GroundedAnswerService",
        FakeGroundedAnswerService,
    )
    monkeypatch.setattr(
        cli,
        "FusedRetrievalPipeline",
        FakeFusedRetrievalPipeline,
    )
    monkeypatch.setattr(
        cli,
        "create_grounded_answer_chain",
        fake_create_grounded_answer_chain,
    )

    exit_code = cli.main(
        [
            "answer",
            str(pdf_path),
            "What did the experiment show?",
            "--top-k",
            "4",
            "--text-model",
            "text-model",
            "--visual-model",
            "visual-model",
            "--answer-model",
            "vision-model",
            "--device",
            "cpu",
            "--text-batch-size",
            "8",
            "--visual-batch-size",
            "6",
            "--chunk-size",
            "120",
            "--overlap",
            "20",
            "--visual-dpi",
            "180",
            "--image-detail",
            "low",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert output == ("Answer: The experiment showed improved performance.\nCitations: pages 2\n")
    assert captured["pdf_path"] == pdf_path
    assert captured["answer_model"] == "vision-model"
    assert captured["image_detail"] == "low"
    assert captured["pipeline_configuration"] == {
        "chunk_size": 120,
        "overlap": 20,
        "visual_dpi": 180,
    }
    assert captured["request"] == AnswerRequest(
        question="What did the experiment show?",
        top_k=4,
    )
    assert captured["pipeline_entered"] is True
    assert captured["pipeline_exited"] is True


def test_answer_command_rejects_large_overlap(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "answer",
                str(tmp_path / "paper.pdf"),
                "Question?",
                "--chunk-size",
                "10",
                "--overlap",
                "10",
            ]
        )

    assert error.value.code == 2
    assert "overlap must be smaller than chunk-size" in capsys.readouterr().err
