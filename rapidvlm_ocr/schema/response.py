# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import ModelName, TaskType


@dataclass(slots=True)
class InferenceResponse:
    task_type: TaskType
    model_name: ModelName
    raw_output: str
    parsed_output: Any
    elapse: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
