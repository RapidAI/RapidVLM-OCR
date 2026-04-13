# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

import time

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

    def generate(self, request: InferenceRequest) -> str:
        sampling_params = SamplingParams(**(request.generation_config or {}))
        response = self.model.generate(
            {
                "prompt": request.prompt,
                "multi_modal_data": {"image": request.image},
            },
            sampling_params=sampling_params,
        )
        return response[0].outputs[0].text
