# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from abc import ABC, abstractmethod
from typing import List

from ..schema.request import InferenceRequest


class BaseEngine(ABC):
    @abstractmethod
    def generate(self, request: InferenceRequest) -> str:
        pass

    @abstractmethod
    def generate_batch(self, requests: List[InferenceRequest]) -> List[str]:
        pass
