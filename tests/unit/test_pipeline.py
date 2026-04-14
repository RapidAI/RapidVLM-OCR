# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

import pytest
import requests

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from rapidvlm_ocr.core.pipeline import Pipeline
from rapidvlm_ocr.schema.enums import ModelName, OutputFormat, TaskType
from rapidvlm_ocr.schema.request import InferenceRequest


def test_invalid_task_type():
    class InvalidTaskType(Enum):
        INVALID_TASK = "invalid_task"

    pipeline = Pipeline(engine=None)

    with pytest.raises(
        ValueError,
        match="Failed to build prompt for model=qianfan_ocr, task=invalid_task: 'invalid_task'",
    ):
        pipeline.run(
            request=InferenceRequest(
                task_type=InvalidTaskType.INVALID_TASK,  # type: ignore[arg-type]
                model_name=ModelName.QIANFAN_OCR,
                output_format=OutputFormat.TEXT,
                prompt="",
                image="tests/test_files/QianFan_OCR/document.png",
            )
        )


def test_batch_mixed_task_types():
    pipeline = Pipeline(engine=None)

    requests = [
        InferenceRequest(
            task_type=TaskType.DOCUMENT_PARSING,
            model_name=ModelName.QIANFAN_OCR,
            output_format=OutputFormat.TEXT,
            prompt="",
            image="tests/test_files/QianFan_OCR/document.png",
        ),
        InferenceRequest(
            task_type=TaskType.TEXT_EXTRACTION,
            model_name=ModelName.QIANFAN_OCR,
            output_format=OutputFormat.JSON,
            prompt="",
            image="tests/test_files/QianFan_OCR/general.jpg",
        ),
    ]

    with pytest.raises(
        ValueError, match="All requests in a batch must have the same task type."
    ):
        pipeline.run_batch(requests)


def test_batch_mixed_model_names():
    class InvalidModelName(Enum):
        INVALID_MODEL = "invalid_model"

    pipeline = Pipeline(engine=None)

    requests = [
        InferenceRequest(
            task_type=TaskType.DOCUMENT_PARSING,
            model_name=ModelName.QIANFAN_OCR,
            output_format=OutputFormat.TEXT,
            prompt="",
            image="tests/test_files/QianFan_OCR/document.png",
        ),
        InferenceRequest(
            task_type=TaskType.DOCUMENT_PARSING,
            model_name=InvalidModelName.INVALID_MODEL,  # type: ignore[arg-type]
            output_format=OutputFormat.TEXT,
            prompt="",
            image="tests/test_files/QianFan_OCR/general.jpg",
        ),
    ]

    with pytest.raises(
        ValueError, match="All requests in a batch must have the same model name."
    ):
        pipeline.run_batch(requests)


def test_run_batch_raises_when_engine_output_count_mismatches_requests():
    class FakeEngine:
        def generate(self, request: InferenceRequest):
            return "unused"

        def generate_batch(self, requests: list[InferenceRequest]):
            return ["output1"]

    pipeline = Pipeline(engine=FakeEngine())

    requests = [
        InferenceRequest(
            task_type=TaskType.DOCUMENT_PARSING,
            model_name=ModelName.QIANFAN_OCR,
            output_format=OutputFormat.TEXT,
            prompt="",
            image="tests/test_files/QianFan_OCR/document.png",
        ),
        InferenceRequest(
            task_type=TaskType.DOCUMENT_PARSING,
            model_name=ModelName.QIANFAN_OCR,
            output_format=OutputFormat.TEXT,
            prompt="",
            image="tests/test_files/QianFan_OCR/general.jpg",
        ),
    ]

    with pytest.raises(RuntimeError, match="Engine returned 1 outputs for 2 requests"):
        pipeline.run_batch(requests)
