# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .enums import ModelName, OutputFormat, TaskType


@dataclass
class InferenceRequest:
    task_type: TaskType
    model_name: ModelName
    output_format: OutputFormat
    prompt: str
    image: Image.Image | str
    generation_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
