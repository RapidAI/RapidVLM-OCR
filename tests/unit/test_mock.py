# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from rapidvlm_ocr import EngineType, InferenceResponse, RapidVLMOCR, TaskType

TEST_FILES_DIR = root_dir / "tests" / "test_files" / "QianFan_OCR"

app = RapidVLMOCR(engine=EngineType.MOCK)


def test_mock_single_image_inference_returns_expected_response() -> None:
    image_path = TEST_FILES_DIR / "general.jpg"

    response = app.run(
        TaskType.TEXT_EXTRACTION, image_path=image_path, metadata={"page": 1}
    )

    assert isinstance(response, InferenceResponse)
    assert response.task_type == TaskType.TEXT_EXTRACTION
    assert response.raw_output == "mock text extraction"
    assert response.parsed_output == "mock text extraction"
    assert response.metadata["image_path"] == str(image_path.resolve())
    assert response.metadata["page"] == 1


def test_mock_batch_inference_returns_expected_responses() -> None:
    image_paths = [TEST_FILES_DIR / "general.jpg", TEST_FILES_DIR / "general_1.jpeg"]

    responses = app.run_batch(
        TaskType.TEXT_EXTRACTION,
        image_paths=image_paths,
        metadata_list=[{"page": 1}, {"page": 2}],
        batch_size=2,
    )

    assert len(responses) == 2
    for idx, response in enumerate(responses, start=1):
        assert isinstance(response, InferenceResponse)
        assert response.task_type == TaskType.TEXT_EXTRACTION
        assert response.raw_output == "mock text extraction"
        assert response.parsed_output == "mock text extraction"
        assert response.metadata["image_path"] == str(image_paths[idx - 1].resolve())
        assert response.metadata["page"] == idx
        assert response.metadata["batch_size"] == 2
        assert "batch_elapse" in response.metadata


def test_mock_document_parse_returns_expected_markdown() -> None:
    image_path = TEST_FILES_DIR / "document.png"

    response = app.run(TaskType.DOCUMENT_PARSING, image_path=image_path)

    assert isinstance(response, InferenceResponse)
    assert response.task_type == TaskType.DOCUMENT_PARSING
    assert response.raw_output == "# Mock Document\n\nThis is a mock markdown output."
    assert "Mock Document" in response.parsed_output
    assert response.metadata["image_path"] == str(image_path.resolve())


def test_mock_kie_returns_expected_json() -> None:
    image_path = TEST_FILES_DIR / "invoice.jpg"

    response = app.run(TaskType.KIE, image_path=image_path)

    assert isinstance(response, InferenceResponse)
    assert response.task_type == TaskType.KIE
    assert response.raw_output
    assert response.parsed_output == {
        "engine": "mock",
        "task": TaskType.KIE.value,
        "fields": {},
    }
    assert response.metadata["image_path"] == str(image_path.resolve())
