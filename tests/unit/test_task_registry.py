# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from rapidvlm_ocr.schema.enums import TaskType
from rapidvlm_ocr.tasks.document_parse import DocumentParseTask
from rapidvlm_ocr.tasks.registry import get_task_handler


class FakeTaskType(str, Enum):
    UNKNOWN = "unknown_task"


def test_get_task_handler_returns_document_parse_task() -> None:
    handler = get_task_handler(TaskType.DOCUMENT_PARSING)

    assert isinstance(handler, DocumentParseTask)


def test_get_task_handler_raises_for_unknown_task() -> None:
    with pytest.raises(KeyError, match="unknown_task"):
        get_task_handler(FakeTaskType.UNKNOWN)  # type: ignore[arg-type]
