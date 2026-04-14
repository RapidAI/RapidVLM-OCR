# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import time

from ..preprocess.image import load_image
from ..preprocess.prompt import PromptBuilder
from ..schema.request import InferenceRequest
from ..schema.response import InferenceResponse
from ..tasks.registry import get_task_handler
from ..utils.logger import logger


class Pipeline:
    def __init__(self, engine):
        self.engine = engine
        self.prompt_builder = PromptBuilder()

    def run(self, request: InferenceRequest) -> InferenceResponse:
        s0 = time.perf_counter()

        runtime_request, image_path = self.prepare_runtime_request(request)
        raw_output = self.engine.generate(runtime_request)

        task_handler = get_task_handler(request.task_type)
        parsed_output = task_handler.postprocess(raw_output)

        elapse = time.perf_counter() - s0
        return InferenceResponse(
            task_type=request.task_type,
            model_name=request.model_name,
            raw_output=raw_output,
            parsed_output=parsed_output,
            metadata={
                "prompt": runtime_request.prompt,
                "image_path": image_path,
                **request.metadata,
            },
            elapse=elapse,
        )

    def run_batch(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        if not requests:
            return []

        self.validate_batch_requests(requests)

        s0 = time.perf_counter()

        runtime_requests = []
        prompts = []
        image_paths = []

        for request in requests:
            runtime_request, image_path = self.prepare_runtime_request(request)
            runtime_requests.append(runtime_request)
            prompts.append(runtime_request.prompt)
            image_paths.append(image_path)

        raw_outputs = self.engine.generate_batch(runtime_requests)
        if len(raw_outputs) != len(requests):
            raise RuntimeError(
                f"Engine returned {len(raw_outputs)} outputs for {len(requests)} requests"
            )

        responses = []
        for request, raw_output, prompt, image_path in zip(
            requests, raw_outputs, prompts, image_paths
        ):
            task_handler = get_task_handler(request.task_type)
            parsed_output = task_handler.postprocess(raw_output)
            response = InferenceResponse(
                task_type=request.task_type,
                model_name=request.model_name,
                raw_output=raw_output,
                parsed_output=parsed_output,
                metadata={
                    "prompt": prompt,
                    "image_path": image_path,
                    **request.metadata,
                },
                elapse=0.0,
            )
            responses.append(response)

        total_elapse = time.perf_counter() - s0
        for response in responses:
            response.metadata["batch_elapse"] = total_elapse
            response.metadata["batch_size"] = len(requests)

        return responses

    def prepare_runtime_request(
        self, request: InferenceRequest
    ) -> tuple[InferenceRequest, str | None]:
        prompt = self.prompt_builder.build(
            request.task_type, request.prompt, request.model_name
        )
        logger.info(f"Built prompt for task {request.task_type}: {prompt}")

        image_path = request.image if isinstance(request.image, str) else None
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
        return runtime_request, image_path

    def validate_batch_requests(self, requests: list[InferenceRequest]) -> None:
        task_types = {request.task_type for request in requests}
        if len(task_types) != 1:
            raise ValueError("All requests in a batch must have the same task type.")

        model_names = {request.model_name for request in requests}
        if len(model_names) != 1:
            raise ValueError("All requests in a batch must have the same model name.")
