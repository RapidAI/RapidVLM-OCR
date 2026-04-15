# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import sys
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from rapidvlm_ocr import EngineType, RapidVLMOCR, TaskType


def test_call_dispatches_to_run_for_single_image(monkeypatch: pytest.MonkeyPatch):
    app = RapidVLMOCR(engine=EngineType.MOCK)

    called = {}

    def fake_run(*args, **kwargs):
        called["run"] = {"args": args, "kwargs": kwargs}
        return "single-result"

    monkeypatch.setattr(app, "run", fake_run)

    result = app(
        task_type=TaskType.DOCUMENT_PARSING,
        input_path="tests/test_files/QianFan_OCR/document.png",
        metadata={"page": 1},
    )

    assert result == "single-result"
    assert "run" in called
    assert called["run"]["kwargs"]["task_type"] == TaskType.DOCUMENT_PARSING
    assert (
        called["run"]["kwargs"]["input_path"]
        == "tests/test_files/QianFan_OCR/document.png"
    )
    assert called["run"]["kwargs"]["metadata"] == {"page": 1}


def test_call_dispatches_single_pdf_to_run_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = RapidVLMOCR(engine=EngineType.MOCK)
    called = {}

    def fake_run_batch(*args, **kwargs):
        called["run_batch"] = {"args": args, "kwargs": kwargs}
        return ["batch-result"]

    monkeypatch.setattr(app, "run_batch", fake_run_batch)

    result = app(
        task_type=TaskType.DOCUMENT_PARSING,
        input_path="tests/test_files/test.pdf",
        metadata={"doc_id": 1},
        batch_size=2,
    )

    assert result == ["batch-result"]
    assert "run_batch" in called
    assert called["run_batch"]["kwargs"]["task_type"] == TaskType.DOCUMENT_PARSING
    assert called["run_batch"]["kwargs"]["input_paths"] == ["tests/test_files/test.pdf"]
    assert called["run_batch"]["kwargs"]["metadata_list"] == [{"doc_id": 1}]
    assert called["run_batch"]["kwargs"]["batch_size"] == 2


def test_call_dispatches_to_run_batch_for_multiple_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = RapidVLMOCR(engine=EngineType.MOCK)
    called = {}

    def fake_run_batch(*args, **kwargs):
        called["run_batch"] = {"args": args, "kwargs": kwargs}
        return ["batch-result"]

    monkeypatch.setattr(app, "run_batch", fake_run_batch)

    result = app(
        task_type=TaskType.TEXT_EXTRACTION,
        input_path=["a.jpg", "b.jpg"],
        metadata={"page": 1},
        batch_size=2,
    )

    assert result == ["batch-result"]
    assert "run_batch" in called
    assert called["run_batch"]["kwargs"]["task_type"] == TaskType.TEXT_EXTRACTION
    assert called["run_batch"]["kwargs"]["input_paths"] == ["a.jpg", "b.jpg"]
    assert called["run_batch"]["kwargs"]["metadata_list"] == [{"page": 1}, {"page": 1}]
    assert called["run_batch"]["kwargs"]["batch_size"] == 2


def test_call_raises_for_list_metadata_with_single_input() -> None:
    app = RapidVLMOCR(engine=EngineType.MOCK)

    with pytest.raises(
        ValueError,
        match="metadata must be a single dict when input_path is a single item",
    ):
        app(
            task_type=TaskType.TEXT_EXTRACTION,
            input_path="a.jpg",
            metadata=[{"page": 1}],
        )
