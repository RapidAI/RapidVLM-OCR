# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import json

from ..schema.enums import EngineType, TaskType
from ..schema.request import InferenceRequest
from .base_engine import BaseEngine


class MockEngine(BaseEngine):
    def __init__(
        self,
        engine_type: EngineType = EngineType.MOCK,
        model_path: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
    ):
        self.engine_type = engine_type
        self.model_path = model_path
        self.base_url = base_url
        self.api_key = api_key

    def generate(self, request: InferenceRequest) -> str:
        if request.task_type == TaskType.OCR:
            return "mock ocr text"

        if request.task_type == TaskType.DOCUMENT_PARSE:
            return "# Mock Document\n\nThis is a mock markdown output."

        if request.task_type == TaskType.KIE:
            return json.dumps(
                {
                    "engine": "mock",
                    "task": request.task_type.value,
                    "fields": {},
                },
                ensure_ascii=False,
            )

        return ""

    def generate_batch(self, requests: list[InferenceRequest]) -> list[str]:
        return [self.generate(request) for request in requests]
