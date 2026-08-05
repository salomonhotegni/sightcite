import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sightcite import cli


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
    ],
)
def test_benchmark_command_rejects_invalid_integer_option(
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
