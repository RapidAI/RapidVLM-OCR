# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from abc import ABC, abstractmethod


from ..schema.request import InferenceRequest


class BaseEngine(ABC):
    @abstractmethod
    def generate(self, request: InferenceRequest) -> str:
        pass
