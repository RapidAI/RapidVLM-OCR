# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from ..config.generation import load_default_generation_config
from ..core.pipeline import Pipeline
from ..preprocess.pdf import load_pdf_images
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
        input_path: str | Path | Sequence[str | Path],
        output_format: OutputFormat | None = None,
        generation_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
        batch_size: int = 4,
    ):
        if isinstance(input_path, (str, Path)):
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(
                    "metadata must be a single dict when input_path is a single item"
                )

            if self.is_pdf_path(input_path):
                return self.run_batch(
                    task_type=task_type,
                    input_paths=[input_path],
                    output_format=output_format,
                    generation_config=generation_config,
                    metadata_list=[dict(metadata or {})],
                    batch_size=batch_size,
                )

            return self.run(
                task_type=task_type,
                input_path=input_path,
                output_format=output_format,
                generation_config=generation_config,
                metadata=metadata,
            )
        elif isinstance(input_path, Sequence):
            input_path = list(input_path)

            if isinstance(metadata, dict) or metadata is None:
                metadata = [dict(metadata or {}) for _ in input_path]

            return self.run_batch(
                task_type=task_type,
                input_paths=input_path,
                output_format=output_format,
                generation_config=generation_config,
                metadata_list=metadata,
                batch_size=batch_size,
            )

        raise ValueError(
            "input_path and metadata must both be either single items or lists"
        )

    def run(
        self,
        task_type: TaskType,
        input_path: str | Path | None = None,
        output_format: OutputFormat | None = None,
        generation_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InferenceResponse:
        if input_path is not None and self.is_pdf_path(input_path):
            raise ValueError("PDF input is only supported via __call__ or run_batch")

        resolved_input_path = (
            str(Path(input_path).resolve()) if input_path is not None else ""
        )
        request = InferenceRequest(
            task_type=task_type,
            model_name=self.model_name,
            prompt="",
            image=resolved_input_path,
            output_format=output_format or self.default_output_format(task_type),
            generation_config=self.merge_generation_config(generation_config),
            metadata={**(metadata or {}), "input_path": resolved_input_path},
        )
        return self.pipeline.run(request)

    def run_batch(
        self,
        task_type: TaskType,
        input_paths: Sequence[str | Path],
        output_format: OutputFormat | None = None,
        generation_config: dict[str, Any] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
        batch_size: int = 4,
    ) -> list[InferenceResponse]:
        if not input_paths:
            return []

        input_paths = list(input_paths)

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if metadata_list is not None and len(metadata_list) != len(input_paths):
            raise ValueError("Length of metadata_list must match length of input_paths")

        generation_config = self.merge_generation_config(generation_config)
        requests = []

        for idx, input_path in enumerate(input_paths):
            metadata = metadata_list[idx] if metadata_list else {}
            for image, expanded_metadata in self.expand_input_path(
                input_path, metadata
            ):
                requests.append(
                    InferenceRequest(
                        task_type=task_type,
                        model_name=self.model_name,
                        prompt="",
                        image=image,
                        output_format=output_format
                        or self.default_output_format(task_type),
                        generation_config=generation_config,
                        metadata=expanded_metadata,
                    )
                )

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

    def is_pdf_path(self, input_path: str | Path) -> bool:
        return Path(input_path).suffix.lower() == ".pdf"

    def expand_input_path(
        self, input_path: str | Path, metadata: dict[str, Any] | None = None
    ) -> list[tuple[str | Image.Image, dict[str, Any]]]:
        resolved_input_path = str(Path(input_path).resolve())
        metadata = dict(metadata or {})

        if not self.is_pdf_path(input_path):
            return [
                (
                    resolved_input_path,
                    {
                        **metadata,
                        "input_path": resolved_input_path,
                    },
                )
            ]

        images = load_pdf_images(resolved_input_path)
        page_count = len(images)
        return [
            (
                image,
                {
                    **metadata,
                    "input_path": resolved_input_path,
                    "page": page_index,
                    "page_count": page_count,
                },
            )
            for page_index, image in enumerate(images, start=1)
        ]
