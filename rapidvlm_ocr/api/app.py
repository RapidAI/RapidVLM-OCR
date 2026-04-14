# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config.generation import load_default_generation_config
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
        default_generation_config: dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self.model_path = str(model_path) if model_path is not None else None

        self.default_generation_config = (
            default_generation_config
            or load_default_generation_config(model_name.value)
        )

        if engine == EngineType.MOCK:
            from ..engine.mock_engine import MockEngine

            self.engine = MockEngine(engine_type=engine, model_path=self.model_path)
        elif engine == EngineType.VLLM:
            from ..engine.vllm_engine import VLLMEngine

            self.engine = VLLMEngine(model_path=self.model_path)

        self.pipeline = Pipeline(self.engine)

    def __call__(
        self,
        task_type: TaskType,
        image_path: str | Path | Sequence[str | Path],
        output_format: OutputFormat | None = None,
        generation_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
        batch_size: int = 4,
    ):
        if isinstance(image_path, (str, Path)):
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(
                    "metadata must be a single dict when image_path is a single item"
                )

            return self.run(
                task_type=task_type,
                image_path=image_path,
                output_format=output_format,
                generation_config=generation_config,
                metadata=metadata,
            )
        elif isinstance(image_path, Sequence):
            image_path = list(image_path)

            if isinstance(metadata, dict) or metadata is None:
                metadata = [dict(metadata or {}) for _ in image_path]

            return self.run_batch(
                task_type=task_type,
                image_paths=image_path,
                output_format=output_format,
                generation_config=generation_config,
                metadata_list=metadata,
                batch_size=batch_size,
            )

        raise ValueError(
            "image_path and metadata must both be either single items or lists"
        )

    def run(
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
            image=str(Path(image_path).resolve()) if image_path is not None else "",
            output_format=output_format or self.default_output_format(task_type),
            generation_config=self.merge_generation_config(generation_config),
            metadata=metadata or {},
        )
        return self.pipeline.run(request)

    def run_batch(
        self,
        task_type: TaskType,
        image_paths: Sequence[str | Path],
        output_format: OutputFormat | None = None,
        generation_config: dict[str, Any] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
        batch_size: int = 4,
    ) -> list[InferenceResponse]:
        if not image_paths:
            return []

        image_paths = list(image_paths)

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if metadata_list is not None and len(metadata_list) != len(image_paths):
            raise ValueError("Length of metadata_list must match length of image_paths")

        generation_config = self.merge_generation_config(generation_config)

        requests = []
        for idx, image_path in enumerate(image_paths):
            metadata = metadata_list[idx] if metadata_list else {}
            request = InferenceRequest(
                task_type=task_type,
                model_name=self.model_name,
                prompt="",
                image=str(Path(image_path).resolve()),
                output_format=output_format or self.default_output_format(task_type),
                generation_config=generation_config,
                metadata=metadata,
            )
            requests.append(request)

        responses = []
        for i in range(0, len(requests), batch_size):
            batch_requests = requests[i : i + batch_size]
            batch_responses = self.pipeline.run_batch(batch_requests)
            responses.extend(batch_responses)
        return responses

    def merge_generation_config(self, generation_config: dict[str, Any] | None):
        merged = dict(self.default_generation_config)
        if generation_config:
            merged.update(generation_config)
        return merged

    def default_output_format(self, task_type: TaskType) -> OutputFormat:
        if task_type == TaskType.DOCUMENT_PARSING:
            return OutputFormat.MARKDOWN

        if task_type == TaskType.KIE:
            return OutputFormat.JSON

        return OutputFormat.TEXT
