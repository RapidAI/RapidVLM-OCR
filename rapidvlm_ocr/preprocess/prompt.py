# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from pathlib import Path

from ..schema.enums import ModelName, TaskType
from ..utils.utils import read_yaml

PROMPT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.yaml"


class PromptBuilder:
    def __init__(self, config_path: str | Path | None = None):
        self.config_path = (
            Path(config_path) if config_path is not None else PROMPT_CONFIG_PATH
        )
        self.prompts = read_yaml(self.config_path)

    def build(
        self, task_type: TaskType, prompt: str | None, model_name: ModelName
    ) -> str:
        if prompt:
            return prompt

        try:
            model_prompts = self.prompts.get(model_name.value)
            prompt_template = model_prompts.get("prompt_template")
            prompt_info = model_prompts["tasks"][task_type.value]["prompt"]
            return prompt_template.format(prompt_info)
        except Exception as e:
            raise ValueError(
                f"Failed to build prompt for model={model_name.value}, task={task_type.value}: {e}"
            ) from e
