# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.pipeline import Pipeline
from ..schema.enums import EngineType, ModelName, OutputFormat, TaskType
from ..schema.request import InferenceRequest
from ..schema.response import InferenceResponse


class RapidVLMOCR:
    def __init__(
        self,
        model_name: ModelName = ModelName.QIANFAN_OCR,
        model_path: str | Path | None = None,
        engine: EngineType = EngineType.MOCK,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        default_generation_config: dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self.model_path = str(model_path) if model_path is not None else None

        self.default_generation_config = default_generation_config or {}

        if engine == EngineType.MOCK:
            from ..engine.mock_engine import MockEngine

            self.engine = MockEngine(
                engine_type=engine,
                model_path=self.model_path,
                base_url=base_url,
                api_key=api_key,
            )
        elif engine == EngineType.VLLM:
            from ..engine.vllm_engine import VLLMEngine

            self.engine = VLLMEngine(model_path=self.model_path)

        self.pipeline = Pipeline(self.engine)

    def __call__(
        self,
        task_type: TaskType,
        image_path: str | Path | None = None,
        output_format: OutputFormat | None = None,
        generation_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InferenceResponse:
        request = InferenceRequest(
            task_type=task_type,
            model_name=self.model_name,
            prompt="",
            image=str(image_path) if image_path is not None else "",
            output_format=output_format or self._default_output_format(task_type),
            generation_config=self._merge_generation_config(generation_config),
            metadata=metadata or {},
        )
        return self.pipeline.run(request)

    def _merge_generation_config(self, generation_config: dict[str, Any] | None):
        merged = dict(self.default_generation_config)
        if generation_config:
            merged.update(generation_config)
        return merged

    def _default_output_format(self, task_type: TaskType) -> OutputFormat:
        if task_type == TaskType.DOCUMENT_PARSE:
            return OutputFormat.MARKDOWN

        if task_type == TaskType.KIE:
            return OutputFormat.JSON

        return OutputFormat.TEXT
