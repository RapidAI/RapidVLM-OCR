# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
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
            return f"[MOCK OCR]"

        if request.task_type == TaskType.DOCUMENT_PARSE:
            return f"[MOCK DOCUMENT_PARSE]"

        if request.task_type == TaskType.KIE:
            schema = request.extras.get("schema") or ["field_1", "field_2"]
            return json.dumps(
                {field: f"mock_{idx}" for idx, field in enumerate(schema, 1)},
                ensure_ascii=False,
            )

        return ""
