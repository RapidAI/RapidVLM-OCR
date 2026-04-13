# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from __future__ import annotations

from ..schema.enums import ModelName, TaskType

PROMPTS = {
    ModelName.QIANFAN_OCR: {
        TaskType.OCR: "Please extract the text from the image.",
        TaskType.DOCUMENT_PARSE: "Parse this document to Markdown.",
        TaskType.KIE: "Please extract key information from the image and output JSON.",
    }
}


class PromptBuilder:
    def build(
        self,
        task_type: TaskType,
        prompt: str | None,
        model_name: ModelName,
    ) -> str:
        if prompt:
            return prompt

        prompt_template = (
            "<|im_start|>user\n<image>\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        return prompt_template.format(PROMPTS[model_name][task_type])
