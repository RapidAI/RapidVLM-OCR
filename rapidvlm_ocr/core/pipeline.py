# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import time

from ..preprocess.image import load_image
from ..preprocess.prompt import PromptBuilder
from ..schema.enums import TaskType
from ..schema.request import InferenceRequest
from ..schema.response import InferenceResponse
from ..tasks.document_parse import DocumentParseTask
from ..tasks.kie import KIETask
from ..tasks.ocr import OCRTask
from ..utils.logger import logger


class Pipeline:
    def __init__(self, engine):
        self.engine = engine
        self.prompt_builder = PromptBuilder()
        self.tasks = {
            TaskType.OCR: OCRTask(),
            TaskType.DOCUMENT_PARSE: DocumentParseTask(),
            TaskType.KIE: KIETask(),
        }

    def run(self, request: InferenceRequest) -> InferenceResponse:
        s0 = time.perf_counter()

        prompt = self.prompt_builder.build(
            request.task_type, request.prompt, request.model_name
        )
        logger.info(f"Built prompt for task {request.task_type}: {prompt}")

        image_path = request.image
        image = (
            load_image(request.image)
            if isinstance(request.image, str)
            else request.image
        )

        runtime_request = InferenceRequest(
            task_type=request.task_type,
            model_name=request.model_name,
            prompt=prompt,
            image=image,
            output_format=request.output_format,
            generation_config=request.generation_config,
            metadata=request.metadata,
        )

        raw_output = self.engine.generate(runtime_request)

        parsed_output = self.tasks[request.task_type].postprocess(raw_output)

        elapse = time.perf_counter() - s0
        return InferenceResponse(
            task_type=request.task_type,
            model_name=request.model_name,
            raw_output=raw_output,
            parsed_output=parsed_output,
            metadata={"prompt": prompt, "image_path": image_path, **request.metadata},
            elapse=elapse,
        )
