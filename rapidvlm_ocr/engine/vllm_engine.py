# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import json
import time
from typing import List

from vllm import LLM, SamplingParams

from ..schema.request import InferenceRequest
from ..utils.logger import logger
from .base_engine import BaseEngine


class VLLMEngine(BaseEngine):
    def __init__(
        self,
        model_path: str | None,
        trust_remote_code: bool = True,
        max_model_len: int = 4096,
        limit_mm_per_prompt: dict[str, int] | None = None,
        **kwargs,
    ):
        if model_path is None:
            raise ValueError("model_path must be provided for VLLMEngine")

        s0 = time.perf_counter()

        logger.info(f"Initializing VLLMEngine with model_path: {model_path}")
        self.model = LLM(
            model=model_path,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            limit_mm_per_prompt=limit_mm_per_prompt,
            **kwargs,
        )
        logger.info("VLLMEngine initialized successfully")
        logger.info(f"Elapse: {time.perf_counter() - s0:.2f} seconds")

    def generate_batch(self, requests: List[InferenceRequest]) -> List[str]:
        sampling_params = self.validate_batch_requests(requests)

        inputs = [self.build_vllm_input(request) for request in requests]
        responses = self.model.generate(inputs, sampling_params=sampling_params)

        outputs = []
        for idx, response in enumerate(responses):
            if not response.outputs:
                logger.warning(f"Empty response for request {idx}: {requests[idx]}")
                outputs.append("")
                continue

            outputs.append(response.outputs[0].text)
        return outputs

    def generate(self, request: InferenceRequest) -> str:
        return self.generate_batch([request])[0]

    def validate_batch_requests(self, requests: List[InferenceRequest]):
        if not requests:
            raise ValueError("Request list is empty")

        config_keys = {self.generate_config_key(request) for request in requests}
        if len(config_keys) > 1:
            raise ValueError(
                "All requests in the batch must have the same generation_config"
            )
        return self.build_sampling_params(requests[0].generation_config)

    def generate_config_key(self, request: InferenceRequest) -> str:
        return json.dumps(request.generation_config, sort_keys=True, ensure_ascii=False)

    def build_sampling_params(self, generation_config: dict) -> SamplingParams:
        return SamplingParams(**generation_config or {})

    def build_vllm_input(self, request: InferenceRequest) -> dict:
        return {
            "prompt": request.prompt,
            "multi_modal_data": {"image": request.image},
        }
