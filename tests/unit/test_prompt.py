# -*- encoding: utf-8 -*-
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from rapidvlm_ocr.preprocess.prompt import PromptBuilder
from rapidvlm_ocr.schema.enums import ModelName, TaskType


class FakeModelName(Enum):
    VALUE = "fake_model"


class FakeTaskType(Enum):
    VALUE = "fake_task"


def test_qianfan_ocr_document_parsing_prompt_is_built_correctly() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        task_type=TaskType.DOCUMENT_PARSING,
        prompt=None,
        model_name=ModelName.QIANFAN_OCR,
    )

    expected = "<|im_start|>user\n<image>\nParse this document to Markdown.<|im_end|>\n<|im_start|>assistant\n"
    assert prompt == expected


def test_prompt_builder_raises_error_for_missing_model() -> None:
    builder = PromptBuilder()

    with pytest.raises(ValueError, match="Failed to build prompt for model=fake_model"):
        builder.build(
            task_type=TaskType.DOCUMENT_PARSING,
            prompt=None,
            model_name=FakeModelName.VALUE,  # type: ignore[arg-type]
        )


def test_prompt_builder_raises_error_for_missing_task() -> None:
    builder = PromptBuilder()

    with pytest.raises(
        ValueError, match="Failed to build prompt for model=qianfan_ocr"
    ):
        builder.build(
            task_type=FakeTaskType.VALUE,  # type: ignore[arg-type]
            prompt=None,
            model_name=ModelName.QIANFAN_OCR,
        )


def test_custom_prompt_is_used_when_provided() -> None:
    builder = PromptBuilder()

    custom_prompt = "This is a custom prompt."
    prompt = builder.build(
        task_type=TaskType.DOCUMENT_PARSING,
        prompt=custom_prompt,
        model_name=ModelName.QIANFAN_OCR,
    )

    assert prompt == custom_prompt
